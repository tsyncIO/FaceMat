"""
Stage 2: Knowledge distillation to student
Reference: Section 4.2 and Algorithm 1
"""
def train_student(config):
    # Load teacher model
    teacher = FaceMatTeacher()
    teacher.load_state_dict(torch.load(config.teacher_ckpt))
    teacher.eval()
    
    # Initialize student
    student = FaceMatStudent()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        for batch in loader:
            img = batch['image'].to(device)
            alpha_gt = batch['alpha'].to(device)
            
            with torch.no_grad():
                _, uncertainty = teacher(img)
            
            # Student prediction
            alpha_pred = student(img)
            
            # Uncertainty-guided loss
            l1_loss = uncertainty_guided_l1(alpha_pred, alpha_gt, uncertainty)
            lap_loss = laplacian_pyramid_loss(alpha_pred, alpha_gt)
            
            total_loss = l1_loss + lap_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # EMA update of teacher (optional)
        if config.use_ema:
            update_teacher_ema(teacher, student)