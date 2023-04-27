        with torch.no_grad():

            for i in range(NUM_ROTATIONS):

                angle = 22.5*i
                
                # Rotate the RGB image
                rotation = iaa.Rotate(angle)
                aug_img = rotation(image=rgb_obs)

                input_tensor = torch.from_numpy(aug_img).to(torch.float32)
                input_tensor /= 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                affordance_map = self.predict(input_tensor)
                affordance_map = torch.clip(affordance_map,0,1)
                
                # Find the best grasp point and angle
                idx = np.argmax(affordance_map)
                N,c,y,x = np.unravel_index(idx, affordance_map.shape)
                curr_val = float(affordance_map[N, c, y, x])

                if curr_val > best_val:
                    best_point = (x, y)
                    best_angle = angle
                    best_val = curr_val

                # Visualize the prediction
                pred_img = torch.squeeze(affordance_map, 0)
                pred_img = pred_img.detach().numpy()
                
                input_img = input_tensor.squeeze(0).detach().numpy()
                vis_img = self.visualize(input_img, pred_img)
                draw_grasp(vis_img, best_point, best_angle)

                imgs.append(vis_img)

        # Rotate the grasp point and angle back to the original image orientation
        
        angle = -best_angle
        
        rotation = iaa.Rotate(angle)

        H, W, _ = rgb_obs.shape
        
        kps = KeypointsOnImage([Keypoint(x=best_point[0], y=best_point[1])], shape=(H, W))
        aug_kps = rotation.augment_keypoints(kps)

        aug_x = int(aug_kps.keypoints[0].x)
        aug_y = int(aug_kps.keypoints[0].y)
        coord = (aug_x, aug_y)

        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        """
        score_map = prediction[0,0].cpu().numpy()
        affordance_map = dict()

        for max_coord in list(self.past_actions):  

            bin = max_coord[0] 
            # supress past actions and select next-best action
            x, y = max_coord[1]
            supression_map = get_gaussian_scoremap(shape=score_map.shape, keypoint=(y,x), sigma=4)
            affordance_map[bin] -= supression_map


        idx = np.argmax(affordance_map)
        y, x = np.unravel_index(idx, affordance_map.shape)
        curr_val = score_map[y,x]
        max_coord = (bin, (x, y))
        self.past_actions.append(max_coord)

        """

        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================

        # Visualize the prediction as a single image

        for img in imgs:
            cv2.line(img, (0, img.shape[0] - 1), (img.shape[1], img.shape[0] - 1), (127, 127, 127), 1)
        
        left_imgs = imgs[:NUM_ROTATIONS // 2]
        right_imgs = imgs[NUM_ROTATIONS // 2:]
        
        left_img = np.concatenate(left_imgs, axis=0).astype(np.uint8)
        right_img = np.concatenate(right_imgs, axis=0).astype(np.uint8)
        
        vis_img = np.concatenate([left_img, right_img], axis=1).astype(np.uint8)

        # ===============================================================================
        return coord, angle, vis_img
