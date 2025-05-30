
import torch
import torch.nn.functional as F
from utils import *
import torch.autograd
import gc
def normalize_img_bc(img):
    """
    Per-image brightness/contrast normalization:
    Subtract mean, divide by std, per sample.
    img: shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    # Flatten the spatial dims for mean/std
    reshaped = img.view(B, C, -1)
    means = reshaped.mean(dim=2, keepdim=True)
    stds  = reshaped.std(dim=2,  keepdim=True) + 1e-8
    normalized = (reshaped - means) / stds
    return normalized.view(B, C, H, W)


def compute_feature_loss(x1, x2, vgg_model, normalize_brightness_contrast=True):
    """
    Computes a feature-based distance using the sum of L2 differences in selected VGG layers.
    Optionally normalizes brightness and contrast for each image before passing to VGG.
    """
    # Optional brightness/contrast normalization:
    if normalize_brightness_contrast:
        x1 = normalize_img_bc(x1)
        x2 = normalize_img_bc(x2)

    # Extract features
    feats1 = vgg_model(x1)
    feats2 = vgg_model(x2)

    # Compute MSE in each layer, average across layers
    dist = 0.0
    for f1, f2 in zip(feats1, feats2):
        # f1, f2 shape is [B, C, H, W]; typically B=1 in an interpolation scenario
        #  -- If you do multi-batch, this still works the same, N_j = B*C*H*W
        #sum_sq = torch.nn.functional.l1_loss(f1, f2, reduction='sum')
        sum_sq = F.mse_loss(f1, f2, reduction='sum')  # sum of (f1 - f2)^2
        N_j = f1.numel()  # total number of scalars in that feature map
        layer_dist = sum_sq / float(N_j)
        dist += layer_dist

    return dist


def perceptual_distance(x1, x2, perceptual_model, normalize_brightness_contrast=True):
    """
    Computes a feature-based distance using the sum of L2 differences in selected VGG layers.
    Optionally normalizes brightness and contrast for each image before passing to VGG.
    """
    # Optional brightness/contrast normalization:
    x1 = (x1 + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x1 = vgg_preprocessing_256(x1)  # Apply preprocessing for VGG]
    x2 = (x2 + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x2 = vgg_preprocessing_256(x2)  # Apply preprocessing for VGG]
    
    if normalize_brightness_contrast:
        x1 = normalize_img_bc(x1)
        x2 = normalize_img_bc(x2)

    # Extract features
    feats1 = perceptual_model(x1)
    feats2 = perceptual_model(x2)

    # Compute MSE in each layer, average across layers
    dist = 0.0
    for f1, f2 in zip(feats1, feats2):
        # f1, f2 shape is [B, C, H, W]; typically B=1 in an interpolation scenario
        #  -- If you do multi-batch, this still works the same, N_j = B*C*H*W
        #sum_sq = torch.nn.functional.l1_loss(f1, f2, reduction='sum')
        sum_sq = F.mse_loss(f1, f2, reduction='sum')  # sum of (f1 - f2)^2
        N_j = f1.numel()  # total number of scalars in that feature map
        layer_dist = sum_sq / float(N_j)
        dist += layer_dist
        
    return dist
    

def feature_energy(path, generator, perceptual_model, is_latent):
    """
    path: tensor shape [t+1, latent_dim].
          path[0] is z0 (fixed or not, depending on your design).
    
    We sum over segments i=0..t-1:
        norm( Jv(z_i, z_{i+1}-z_i) )^2
      = < (z_{i+1}-z_i),  G(z_i)  (z_{i+1}-z_i) >
      where G(z_i) = J(z_i)^T J(z_i).

    We'll use matrix-free approach:
       If r_i = z_{i+1}-z_i,
       then segment energy = || Jv(z_i, r_i) ||^2.

    Returns: a scalar (PyTorch tensor) that can be backprop'ed.
    """
    total_energy = 0.0
    t = path.shape[0] - 1
    dt = 1/t
    
    for i in range(t):
        #z_i = path[i]     # shape [latent_dim]
        #z_ip1 = path[i+1]
        xi,_ = generator([path[i] ], input_is_latent=is_latent, randomize_noise=False)
        xip1,_ = generator([path[i+1]], input_is_latent=is_latent, randomize_noise=False)
        #diff = z_ip1 - z_i
        # Jv => shape [3,32,32]
        #u = Jv(generator, z_i, diff, create_graph=True)
        u = perceptual_distance(xi, xip1, perceptual_model)
        seg_energy = 0.5*dt*(u ** 2).sum()  # scalar
        total_energy = total_energy + seg_energy
    
    return total_energy

def classifier_loss(generator, classifier, z_end, target_label):
    """
    Simple cross-entropy at the final latent code z_end.
    target_label: int in [0..9].
    """
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)  # => [1,latent_dim]
        
    x_end,_ = generator([z_end], input_is_latent=False, randomize_noise=False)  # shape: [1, 3, 256, 256]
    x_end = (x_end + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    
    #x = (x + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x_end = vgg_preprocessing(x_end)  # Apply preprocessing for VGG]
    
    logits = classifier(x_end)      # => [1,10]
    target = torch.tensor([target_label], dtype=torch.long, device=z_end.device)
    #print("logits.shape =", logits.shape)  # e.g. [1, 10] or [1, 1000], etc.
    #print("target_label =", target_label)
    loss_val = F.cross_entropy(logits, target)
    return loss_val
    

def full_path_loss(path, generator, classifier, perceptual_model, target_label, is_latent=cfg.is_latent, lam=1.0):
    """
    Combine discrete geodesic energy + lam * classifier loss on final point.
    """
    geo = feature_energy(path, generator, perceptual_model, is_latent)
    clf = classifier_loss(generator, classifier, path[-1], target_label)
    
    return geo + lam * clf

def discrete_geodesic_energy(path, generator, is_latent):

    total_energy = 0.0
    t = path.shape[0] - 1
    
    for i in range(t):
        z_i = path[i]     # shape [latent_dim]
        z_ip1 = path[i+1]
        diff = z_ip1 - z_i
        # Jv => shape [3,32,32]
        u = Jv(generator, z_i, diff, is_latent, create_graph=True)
        seg_energy = 0.5*(u ** 2).sum()  # scalar
        total_energy = total_energy + seg_energy
    
    return total_energy


def geodesic_only(path_init, generator, perceptual_model, is_latent, steps=30, lr=1e-2):
    """
    Phase 1: only minimize the discrete geodesic energy
    for the path (ignoring classifier).
    
    path_init: shape [t+1, latent_dim]
    We'll keep path_init[0] fixed, treat path_init[1..t] as trainable.
    """
    zstart = path_init[0].detach().clone()  # fixed
    zend = path_init[-1].detach().clone() 
    z_vars = nn.Parameter(path_init[1:-1].clone())  # shape [t, latent_dim]
    
    opt = torch.optim.Adam([z_vars], lr=lr)
    
    for step_idx in range(steps):
        opt.zero_grad()
        # Rebuild the full path
        current_path = torch.cat([zstart.unsqueeze(0), z_vars, zend.unsqueeze(0)], dim=0)  # shape [t+1, latent_dim]

        if perceptual_model==None:
            #geo_energy = discrete_energy_no_jacobian(current_path, generator, is_latent)
            geo_energy = discrete_geodesic_energy(current_path, generator, is_latent)

        else:
            geo_energy = feature_energy(current_path, generator, perceptual_model, is_latent)
        #geo_energy = discrete_geodesic_energy(current_path, generator, is_latent)
        #geo_energy = discrete_energy_no_jacobian(current_path, generator, is_latent)
        geo_energy.backward()  # backprop
        
        opt.step()
        
        if (step_idx+1) % 10 == 0:
            print(f"[Phase1] step {step_idx+1}/{steps}, geodesic_energy={geo_energy.item():.4f}")
    
    final_path = torch.cat([zstart.unsqueeze(0), z_vars.detach(), zend.unsqueeze(0)], dim=0)
    return final_path

def phase2_add_classifier(
    path_init, 
    generator, 
    classifier,
    perceptual_model,
    is_latent,
    target_label,
    lam=10.0,
    steps=500,
    lr=1e-2, 
    lam_increase_interval=10,
    lam_increase_factor=5,
    dynamics_list=None
):
    """
    Phase 2: from the path_init (already partially geodesic),
    now add classifier constraint on the endpoint. 
    Optimize:
        discrete_geodesic_energy + lam * cross_entropy(classifier(g(z_T)), target_label).
    """
    z0 = path_init[0].detach().clone()  # keep it fixed
    
    initial_path = torch.cat([z0.unsqueeze(0), path_init[1:].detach()], dim=0)
    reanchored = reanchor_path(z0, initial_path, generator, classifier, target_label, is_latent)
    z_vars = nn.Parameter(reanchored.clone())
        
    #z_vars = nn.Parameter(path_init[1:].clone())
    current_lam = lam
    classifier.eval()
    opt = torch.optim.Adam([z_vars], lr=lr)
    
    for step_idx in range(steps):
        if step_idx > 0 and (step_idx % lam_increase_interval == 0):
            current_lam *= lam_increase_factor
            
            current_path = torch.cat([z0.unsqueeze(0), z_vars.detach()], dim=0)
            z_vars.data = reanchor_path(z0, current_path, generator, classifier, target_label, is_latent).clone()
    
            print(f"Step {step_idx}: Increasing lam to {current_lam:.4f} and reanchoring endpoint")
            dynamics_list.append(torch.cat([z0.unsqueeze(0), z_vars.detach()], dim=0))
            #print(f"Step {step_idx}: Increasing lam to {current_lam:.4f}")
        opt.zero_grad()
        current_path = torch.cat([z0.unsqueeze(0), z_vars], dim=0)
        
        loss_val = full_path_loss(current_path, generator, classifier, perceptual_model, target_label, is_latent, lam=current_lam, step_idx=step_idx)
        loss_val.backward()
        opt.step()
        
        if (step_idx+1) % 10 == 0:
            # Just to monitor progress, we can check the separate terms
            with torch.no_grad():
                geo_val = feature_energy(current_path, generator, perceptual_model, is_latent)
                clf_val = classifier_loss(generator, classifier, current_path[-1], target_label)
            print(f"[Phase2] step {step_idx+1}/{steps}, total_loss={loss_val.item():.4f}, "
                  f"geo={geo_val.item():.4f}, clf={clf_val.item():.4f}")
    
    final_path = torch.cat([z0.unsqueeze(0), z_vars.detach()], dim=0)
    return final_path

def reanchor_path(z0, current_path, generator, classifier, target_label, is_latent):
    """
    From current_path, find the first point classified as target_label,
    and return interpolated path to it from z0 with same number of steps.
    """
    with torch.no_grad():
        path = current_path.detach()
        all_images = []
        for i in range(len(path)):
            # Get the final image for the current latent vector
            x_final, _ = generator([path[i]], input_is_latent=is_latent, randomize_noise=False)
            x_final = (x_final + 1) / 2  
            x_final = vgg_preprocessing(x_final)  
            all_images.append(x_final)

    
        gens = torch.cat(all_images, dim=0)
        
        logits = classifier(gens)
        preds = logits.argmax(dim=1)
        target_idxs = (preds == target_label).nonzero(as_tuple=False)
    
        if len(target_idxs) > 0:
            target_idx = target_idxs[0].item()  # first occurrence
            new_endpoint = path[target_idx]
            # Create interpolated path of same length
            steps = path.size(0)
            interpolated = [z0 + float(i)/(steps-1)*(new_endpoint - z0) for i in range(1, steps)]
            return torch.stack(interpolated, dim=0)  # skip z0, we keep it fixed
        else:
            return current_path[1:]  # fallback: no change



#########################################################
# Robust feature helper functions (as provided)        #
#########################################################

def robust_feats_fn(x, robust_feats):
    """
    Given an image tensor x, compute the robust features.
    Assumes robust_feats(x) returns an iterable of feature maps.
    """
    feats = robust_feats(x)
    return tuple(feats)

def M_f_xv(x, v, robust_feats):
    """
    Given an image x and a perturbation v (both with gradients), compute
         M_f(x) v = sum_i J_{f_i}(x)^T [J_{f_i}(x) v],
    where the sum runs over the different feature maps f_i returned by robust_feats.
    """
    if not x.requires_grad:
        x.requires_grad_()
    # Compute the JVP through robust_feats_fn.
    feats, jvp_tuple = torch.autograd.functional.jvp(
        lambda x_: robust_feats_fn(x_, robust_feats),
        x, v, create_graph=True)
    
    result = 0
    # For each feature map f_i, compute the VJP that “pulls back” the directional derivative.
    for i in range(len(feats)):
        # The VJP: grad( f_i(x), x, grad_outputs = (jvp of f_i) )
        res_i = torch.autograd.grad(feats[i], x, grad_outputs=jvp_tuple[i],
                                    retain_graph=True, create_graph=True)[0]
        result = result + res_i
        # Optionally detach and delete temporary tensors:
        res_i.detach()
        del res_i
    del feats, jvp_tuple
    return result

#########################################################
# Jacobian-vector and transpose helpers for the generator #
#########################################################

def g_wrap(z, generator):
    """
    Wrap the generator so that it returns the generated image tensor.
    Assumes generator returns a list/tuple and we use the first element.
    """
    out = generator([z], input_is_latent=False, randomize_noise=False)
    return out[0]

def Jv(generator, z, v):
    """
    Compute the Jacobian-vector product:  J_g(z) * v,
    where J_g(z) = d g(z)/dz.
    
    Uses torch.autograd.functional.jvp.
    
    Arguments:
      generator: the generator network.
      z: latent code tensor of shape [latent_dim].
      v: direction tensor of the same shape as z.
    
    Returns:
      Tensor with the same shape as g_wrap(z, generator), e.g. [C,H,W].
    """
    z = z.detach().requires_grad_(True)
    jvp_out = torch.autograd.functional.jvp(
        func=lambda z_: g_wrap(z_, generator),
        inputs=z,
        v=v,
        create_graph=False
    )[1]
    return jvp_out

def JT(u, generator, z):
    """
    Compute the Jacobian-transpose vector product:  J_g(z)^T * u,
    where u has the same shape as g_wrap(z, generator) (e.g. [C,H,W]).
    
    Implements the reverse-mode trick:
      1. Compute x = g_wrap(z, generator).
      2. Let dummy = (x * u).sum().
      3. Then grad(dummy, z) = J_g(z)^T * u.
    
    Returns:
      Tensor with the same shape as z.
    """
    z = z.detach().requires_grad_(True)
    x = g_wrap(z, generator)
    dummy = (x * u).sum()
    grad_z = torch.autograd.grad(dummy, z, create_graph=False, retain_graph=False)[0]
    return grad_z

#########################################################
# Our combined Mv operator using robust metric         #
#########################################################

def Mv(generator, z, w, robust_feats):
    """
    Compute the product:
       M(z) w = J_g(z)^T [ M_f( g(z) ) ( J_g(z)*w ) ]
    where:
      - J_g(z) is the Jacobian of the generator at z.
      - M_f(x) is the robust feature metric computed via M_f_xv.
    
    Arguments:
      generator: the generator network.
      z: current latent code, tensor of shape [latent_dim].
      w: a perturbation in latent space (same shape as z).
      robust_feats: robust feature extractor module.
    
    Returns:
      Tensor of shape [latent_dim] representing M(z)w.
    """
    # (1) Compute v' = J_g(z) * w.
    v_prime = Jv(generator, z, w)
    
    # (2) Compute x = g_wrap(z, generator)
    x = g_wrap(z, generator)
    
    # (3) Compute robust metric product: m_f_v = M_f( x ) (v_prime)
    m_f_v = M_f_xv(x, v_prime, robust_feats)
    
    # (4) Finally, compute J_g(z)^T [m_f_v]
    return JT(m_f_v, generator, z)

#########################################################
# Matrix-free Conjugate Gradient Solver using Mv       #
#########################################################

def conjugate_gradient(Mv_fn, b, cg_iters=15, residual_tol=1e-3):
    """
    Solve M x = b using a matrix-free conjugate gradient method,
    where the operator M is provided as a function Mv_fn.
    
    Arguments:
      Mv_fn: function that computes M(w) given a vector w.
      b: right-hand side tensor.
      cg_iters: maximum number of iterations.
      residual_tol: stopping tolerance on the residual norm.
    
    Returns:
      x: approximate solution tensor.
    """
    x = torch.zeros_like(b)
    r = b - Mv_fn(x)
    p = r.clone()
    rsold = torch.dot(r.flatten(), r.flatten())
    
    for i in range(cg_iters):
        Ap = Mv_fn(p)
        p_dot_Ap = torch.dot(p.flatten(), Ap.flatten())
        if torch.abs(p_dot_Ap) < 1e-10:
            break
        alpha = rsold / p_dot_Ap
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

#########################################################
# Phase 2 Optimization incorporating the robust metric   #
#########################################################

def phase2_add_classifier_G(path_init, generator, classifier, vgg_model,
                          is_latent, target_label, lam=1e-6, steps=700, lr=1e-2,
                          lam_increase_interval=80, lam_increase_factor=5, cg_steps=15):
    """
    Phase 2 optimization: minimize a geodesic energy plus a classifier loss.
    For the classifier loss term we “naturalize” the gradient by replacing the
    Euclidean gradient grad_clf with a natural gradient correction r that satisfies:
    
         M(z_end) r = grad_clf,
    
    where the metric is:
         M(z_end) = J_g(z_end)^T [ M_f(g(z_end)) (J_g(z_end) *) ],
    
    with M_f computed via robust features.
    
    Arguments:
      path_init: tensor of latent codes, shape [num_points, latent_dim]
      generator: the generator network.
      classifier: classifier network (used inside classifier_loss).
      robust_feats: robust feature extractor module.
      is_latent: flag (passed to feature_energy and classifier_loss).
      target_label: target label for the classifier.
      lam: base weighting for the classifier term.
      steps: number of optimization steps.
      lr: learning rate.
      lam_increase_interval, lam_increase_factor: schedule for λ.
      cg_steps: maximum number of conjugate gradient iterations.
    
    Returns:
      final_path: optimized latent code path.
    """
    # Fix the starting latent code.
    z0 = path_init[0].detach().clone()
    # Optimize the remaining latent codes.
    z_vars = nn.Parameter(path_init[1:].clone())
    current_lam = lam
    opt = torch.optim.Adam([z_vars], lr=lr)
    
    for step_idx in range(steps):
        if step_idx > 0 and (step_idx % lam_increase_interval == 0):
            current_lam *= lam_increase_factor
            print(f"Step {step_idx}: Increasing lam to {current_lam:.4f}")
        
        opt.zero_grad()
        # Rebuild the full path: fixed start + trainable latent codes.
        current_path = torch.cat([z0.unsqueeze(0), z_vars], dim=0)
        
        # (A) Compute geodesic (feature) energy.
        geo_loss = feature_energy(current_path, generator, vgg_model, is_latent)
        geo_loss.backward(retain_graph=False)

        grad_geo = z_vars.grad[-1].detach().clone()
        
        # (B) Compute classifier loss at the final latent code.
        z_final = current_path[-1].detach().clone().requires_grad_(True)
        clf_loss = classifier_loss(generator, classifier, z_final, target_label)
        grad_clf = torch.autograd.grad(clf_loss, z_final, create_graph=False)[0]
        
        # (C) Define an Mv function that uses the current z_final and robust_feats.
        def Mv_fn(w):
            # w is a tensor with the same shape as z_final.
            return Mv(generator, z_final, w, vgg_model)
        
        # (D) Solve M r = grad_clf using conjugate gradient.
        #natural_grad = conjugate_gradient(Mv_fn, grad_clf, cg_iters=cg_steps)
        #natural_grad_norm = natural_grad.norm(p=2) + 1e-12
        #natural_grad = natural_grad / natural_grad_norm
        total_grad = grad_geo + current_lam * grad_clf
        natural_grad = conjugate_gradient(Mv_fn, total_grad, cg_iters=cg_steps)
        natural_grad_norm = natural_grad.norm(p=2) + 1e-12
        natural_grad = natural_grad / natural_grad_norm
        # (E) Add the natural gradient contribution (scaled by current_lam) to the gradient
        # of the final latent code (which is stored inside z_vars).
        if z_vars.grad is None:
            z_vars.grad = torch.zeros_like(z_vars)
        #z_vars.grad[-1].add_(current_lam * natural_grad)
        z_vars.grad[-1].copy_(natural_grad)
        
        total_loss = geo_loss + current_lam * clf_loss
        if (step_idx + 1) % 10 == 0:
            with torch.no_grad():
                geo_val = feature_energy(current_path, generator, vgg_model, is_latent)
                clf_val = classifier_loss(generator, classifier, current_path[-1], target_label)
            print(f"[Phase2] step {step_idx+1}/{steps}, total_loss={total_loss.item():.4f}, "
                  f"geo={geo_val.item():.4f}, clf={clf_val.item():.4f}")
            del geo_val, clf_val
        
        opt.step()
        
        # Clean up temporary tensors.
        del geo_loss, clf_loss, grad_clf, natural_grad, current_path, total_loss
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
            
    final_path = torch.cat([z0.unsqueeze(0), z_vars.detach()], dim=0)
    return final_path