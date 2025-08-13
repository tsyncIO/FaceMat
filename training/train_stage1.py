"""
Stage 1: Teacher model training
Reference: Section 4.1 and Algorithm 1
"""
def train_teacher(config):
    # Initialize dataset
    transform = MattingTransform(config.input_size)
    dataset = CelebAMatDataset(
        config.face_dir, 
        config.occ_dir,
        transform=transform,
        size=config.dataset_size
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model
    model = FaceMatTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        for batch in loader:
            img = batch['image'].to(device)
            alpha_gt = batch['alpha'].to(device)
            trimap = batch['trimap'].to(device)
            
            # Create unknown region mask
            mask = (trimap == 0.5).float()
            
            # Forward pass
            alpha_pred, var_pred = model(img)
            
            # Calculate losses
            l1_loss = masked_l1(alpha_pred, alpha_gt, mask)
            lap_loss = laplacian_pyramid_loss(alpha_pred, alpha_gt)
            nll_alpha = nll_loss(alpha_pred, alpha_gt, var_pred)
            nll_uncertainty = nll_loss(var_pred, (alpha_pred - alpha_gt).abs().detach(), var_pred)
            
            total_loss = l1_loss + lap_loss + nll_alpha + nll_uncertainty
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), f"teacher_epoch_{epoch}.pth")