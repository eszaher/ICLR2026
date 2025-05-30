import torch
from utils import *
def Jv(generator, z, v):
    """
    Compute the Jacobian-vector product J(z) * v
    where J(z) = d g(z) / d z, and v is the direction in latent space.
    
    Return shape = same as generator(z) => e.g. [3,32,32].
    
    Implementation approach:
     1) We'll do a forward pass to get g(z).
     2) We'll compute the gradient of g(z) wrt. z, but we only want
        the product with v, not the full Jacobian. This can be done
        by treating 'v' as a 'virtual' gradient from the output side.
    """
    # Ensure z and v are both differentiable:
    z = z.detach().requires_grad_(True)
    
    # Forward pass
    x = generator([z], input_is_latent=False, randomize_noise=False) # shape [3,32,32]
    
    # We'll flatten x so that we can feed a single 'vector' as grad_outputs
    # but we can keep x in shape [3,32,32] if we prefer, as long as 
    # we do a matching shape for grad_outputs.
    x_flat = x[0].view(-1)  # shape [3*32*32]
    
    # Now compute gradient of x_flat wrt. z, weighted by v.
    # However, we can't directly pass `v` as grad_outputs because
    # x_flat is length 3072, but v is dimension 32. 
    # Instead, we do a trick: we compute the directional derivative by
    #   (d x_flat / d z) * v
    # using an internal function:
    
    # The typical trick:
    #   If we want Jv = d(g)/dz * v, we can do a gradient w.r.t. z
    #   by specifying 'grad_outputs' that is the result of the chain rule.
    #
    # But we can't directly multiply x by v. Instead, we do:
    #   let dummy = (x_flat * some_vector).sum()
    #   then grad(dummy, z) = (some_vector^T * J) by chain rule.
    #
    # So if 'some_vector' = Jv on the output side, we get J^T * Jv on z.
    # We want just J * v. So we do a second step:
    #
    # Actually, for Jv, we do forward-mode or we can do a 2-step approach 
    # with double backward. 
    #
    # However, PyTorch does not have a built-in forward-mode. We'll do a typical 
    # workaround using 'v' as the gradient wrt. z. See next function for J^T.
    
    # For demonstration, let's do a 'manual' approach with an extra backward pass:
    # 1) We pick a dummy gradient wrt. x that is the same shape as x
    # 2) We use 'torch.autograd.grad' with that dummy gradient 
    #    to get the result wrt. z, which is J^T * dummy.
    # This is for J^T. For Jv in forward-mode, we need a different trick.
    #
    # In practice, we often do the J^T * vector approach (see 'JT' below).
    # Let's show how to do Jv (forward-mode) with "torch.autograd.functional.jvp".
    
    # => If your PyTorch version >= 1.10, you can do:
    # jvp_result = torch.autograd.functional.jvp(
    #     func=generator,
    #     inputs=z,
    #     v=v,
    # )[1]
    # That returns Jv directly. Let's do that:

    jvp_result = torch.autograd.functional.jvp(
        func=g_wrap,
        inputs=z,
        v=v,
        create_graph=False
    )[1]
    # shape of jvp_result: same as g(z) => [3,32,32]

    return jvp_result.squeeze(dim=0)

def JT(u, generator, z):
    """
    Compute J(z)^T * u, 
    where u has shape [3,32,32] = the dimension of the generator output.
    
    Return shape = [latent_dim], e.g. [32].
    
    Implementation approach: standard reverse-mode:
     1) Let dummy = (u * x).sum(), where x=g(z)
     2) Then grad(dummy, z) = sum over all output elems of partial x_i / partial z_j * u_i
       = J(z)^T * u_flat
    """
    # Ensure z is differentiable
    z = z.detach().requires_grad_(True)
    
    # Forward pass
    x = generator([z], input_is_latent=False, randomize_noise=False)  # shape [3,32,32]
    x = x[0].squeeze(dim=0)
    
    # Multiply x by u (elementwise), then sum => shape []
    # But to ensure shape matching, let's do it carefully:
    # x, u both [3,32,32]. So (x*u).sum() is scalar.
    dummy = (x * u).sum()
    
    # Now compute grad(dummy, z)
    grad_z = torch.autograd.grad(
        outputs=dummy, 
        inputs=z,
        create_graph=False,
        retain_graph=False
    )[0]  # shape [latent_dim]
    
    return grad_z.squeeze(dim=0)


def conjugate_gradient_solve(z, generator, b, max_iter=20, tol=1e-5):
    """
    Solve (J^T J) r = b using matrix-free conjugate gradient.
    A(r) = (J^T J) r
    
    z: current latent code, shape [latent_dim]
    generator: the generator network
    b: the right-hand side vector, shape [latent_dim]
    max_iter: maximum CG iterations
    tol: tolerance for early stopping
    
    Returns: r, approximate solution to (J^T J) r = b
    """
    # We'll define a helper function A_times(r) = J^T( J(r) )
    def A_times(r):
        r = r.unsqueeze(dim=0)
        # 1) compute Jv = J(r)
        Jr = Jv(generator, z, r)  # shape [3,32,32]
        # 2) compute J^T( Jr )
        return JT(Jr, generator, z)  # shape [latent_dim]
    
    # Initialize
    r_est = torch.zeros_like(b)      # our current guess
    residual = b.clone()            # residual = b - A_times(r_est) = b - 0 = b
    p = residual.clone()
    rr_old = torch.dot(residual, residual)
    #p = p.unsqueeze(dim=0) # this
    
    for i in range(max_iter):
        Ap = A_times(p)
        alpha = rr_old / torch.dot(p, Ap)
        
        r_est = r_est + alpha * p
        residual = residual - alpha * Ap
        rr_new = torch.dot(residual, residual)
        
        if rr_new.sqrt() < tol:
            break
        
        beta = rr_new / rr_old
        p = residual + beta * p
        rr_old = rr_new
    
    return r_est

def g_wrap(z, generator):
    out = generator([z], input_is_latent=False, randomize_noise=False)
    return out[0]

def riemannian_update_step(z, generator, classifier, target_label, lr=1e-2):
    """
    Perform ONE Riemannian gradient step for binary classification:
      z' = z - lr * ( M(z)^{-1} grad_z f_y(z) ) / || ... ||
    where M(z) = J^T J, and f_y(z) is the binary cross-entropy loss for the classifier.
    
    Returns the updated z, shape [latent_dim].
    """
    # 1) Compute gradient wrt z: grad_z f_y(z)
    z = z.detach().clone().requires_grad_(True)
    #z = z + 0.1* torch.randn_like(z)

    # Generate an image from the latent vector
    x,_ = generator([z], input_is_latent=False, randomize_noise=False)  # shape: [1, 3, 256, 256]
    x = (x + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    
    #x = (x + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x = vgg_preprocessing(x)  # Apply preprocessing for VGG
    #x = x.unsqueeze(dim=0)
    #x = x.unsqueeze(0)  # Add batch dimension, shape: [1, 3, 224, 224]

    # Get the classifier output
    logits = classifier(x)  # shape: [1, 1]
    #logits = logits.squeeze(1)  # Remove the singleton dimension

    # Binary Cross-Entropy Loss
    target = torch.tensor([target_label], dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, target)  # scalar loss
    print(f"Loss: {loss.item():.8f}")
    
    #loss = nn.MSELoss(logits, target) 
    #print("Loss:", loss.item())
    #print("Loss:", loss)

    # Compute gradient wrt z
    grad_z = torch.autograd.grad(loss, z, create_graph=False)[0]  # shape: [latent_dim]
    grad_z = grad_z.squeeze(dim=0)

    # 2) Solve M(z)*r = grad_z for r, i.e. (J^T J) r = grad_z
    #    using conjugate gradient (matrix-free)
    r = conjugate_gradient_solve(z, generator, grad_z, max_iter=25, tol=1e-5)

    # 3) Normalize r
    r_norm = r.norm(p=2) + 1e-12
    r_unit = r / r_norm

    # 4) Update z
    z_updated = z - lr * r_unit
    return z_updated.detach()


def euclidean_update_step(z, generator, classifier, target_label, lr=1e-2):
    """
    Perform ONE Euclidean gradient step for binary classification:
      z' = z - lr * grad_z f_y(z)
    
    Returns the updated z, shape [latent_dim].
    """
    # 1) Compute gradient wrt z: grad_z f_y(z)
    z = z.detach().clone().requires_grad_(True)

    # Generate an image from the latent vector
    x,_ = generator([z], input_is_latent=False, randomize_noise=False)  # shape: [1, 3, 256, 256]
    x = (x + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    
    x = vgg_preprocessing(x)  # Apply preprocessing for VGG

    # Get the classifier output
    logits = classifier(x)  # shape: [1, num_classes]
    
    # Binary Cross-Entropy Loss
    target = torch.tensor([target_label], dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, target)  # scalar loss

    # Compute gradient wrt z
    grad_z = torch.autograd.grad(loss, z, create_graph=False)[0]  # shape: [latent_dim]
    
    # Update z
    z_updated = z - lr * grad_z
    return z_updated.detach()


def revise(z, generator, classifier, target_label, lr=1e-2, lambda_clf=0.01):
    """
    Perform ONE Euclidean gradient step combining classifier and L2 loss:
      z' = z - lr * grad_z ( L_cls(z) + lambda * ||z - z_0||^2 )

    Args:
        z (torch.Tensor): latent vector [latent_dim]
        generator: GAN generator
        classifier: classifier network
        target_label (int): desired class label
        lr (float): learning rate
        lambda_clf (float): weight for clf loss

    Returns:
        z_updated (torch.Tensor): updated latent vector
    """
    z0 = z.detach().clone()  # original z for L2 reference
    z = z.detach().clone().requires_grad_(True)

    # Generate image
    x, _ = generator([z], input_is_latent=False, randomize_noise=False)
    x = (x + 1) / 2  # Scale from [-1, 1] to [0, 1]
    x = vgg_preprocessing(x)

    # Classifier loss
    logits = classifier(x)
    target = torch.tensor([target_label], dtype=torch.long, device=z.device)
    loss_cls = F.cross_entropy(logits, target)

    # L2 regularization loss
    loss_l2 = torch.norm(z - z0, p=2) ** 2  # ||z - z0||^2

    # Combined loss
    total_loss = loss_l2 + lambda_clf * loss_cls 

    # Gradient wrt z
    grad_z = torch.autograd.grad(total_loss, z, create_graph=False)[0]

    # Gradient update
    z_updated = z - lr * grad_z
    return z_updated.detach()