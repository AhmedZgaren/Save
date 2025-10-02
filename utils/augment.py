import random
import numpy as np
import albumentations as A

def clamp_keypoints(keypoints, width, height):
    """Clamps keypoint coordinates to be within image boundaries [0, dim)."""
    clamped_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        clamped_keypoints.append([x, y] + list(kp[2:])) # Keep any extra values
    return clamped_keypoints
def clamp_bbox(bboxes):
    """Clamps bounding box coordinates to be within image boundaries [0, dim)."""
    clamped_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, 1.0))
        y1 = max(0, min(y1, 1.0))
        x2 = max(0, min(x2, 1.0))
        y2 = max(0, min(y2, 1.0))
        clamped_bboxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
    return clamped_bboxes

def random_crop_savebboxes(img,mask,kpoints, bboxes, class_labels,expected_h, expected_w, min_area, min_visibility):
    """
        Implement random crop image
        image_name: for example img_0.jpg
        image_dir: original image directory
        label_dir: orrigianl label directory
        expected_h: expected height of the image, it depends on the scale_y
        expected_w: expected width of the image, it depends on the scale_x
        min_area: threshol of area of bounding boxes. If area of box after augmentation < min_area, we will drop that box.
        min_visibility: [0, 1]  If the ratio of the box area after augmentation to the area of the box before augmentation 
                                becomes smaller than min_visibility, we will drop that box.
    """

    transform = A.Compose([A.RandomCrop(expected_h, expected_w)],
                           bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=min_area, min_visibility=min_visibility, clip=True,filter_invalid_bboxes=True), keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))   # height, width
    bboxes = clamp_bbox(bboxes)  # Ensure bboxes are within image bounds
    transformed = transform(image=img, mask = mask, bboxes=bboxes, class_labels=class_labels, keypoints = kpoints)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']  # list
    transformed_points = transformed['keypoints']
    transformed_points = clamp_keypoints(transformed_points, expected_w, expected_h)

    return transformed_image, transformed_mask,transformed_points, transformed_bboxes, transformed_class_labels

def mosaic(dataset, mo_w, mo_h, scale_x, scale_y, min_area, min_visibility):
    """
        Implement mosaic augmentation
        image_file_list: list of 4 images (only name of image, not path), [img_1.jpg, img_2.jpg, img_3.png, img_4.jpeg]
        image_dir: original image directory
        label_dir: orrigianl label directory
        output_image_dir: path of directory for mosaic images
        output_label_dir: path of directory for new labels
        mo_h: height of mosaic-augmented image
        mo_w: width of mosaic-augmented image
        min_area: threshol of area of bounding boxes. If area of box after augmentation < min_area, we will drop that box.
        min_visibility: [0, 1]  If the ratio of the box area after augmentation to the area of the box before augmentation 
                                becomes smaller than min_visibility, we will drop that box.
    """
    
    #create the image lists
    indices = [random.randint(0, len(dataset) - 1) for _ in range(4)]
    
    imgs,masks, bboxes, all_points,labels= [], [], [], [],[]

    for i in indices:
        sample = dataset[i]
        img, mask, target_boxes, points , label = sample['image'],sample['mask'], sample['boxes'], sample['points'], sample['labels']
        imgs.append(img)
        masks.append(mask)
        bboxes.append(target_boxes)
        all_points.append([(int(pt[0]), int(pt[1])) for pt in points])
        labels.append(label)
   
        
    # creat a new image
    new_img = np.zeros((mo_h, mo_w, 3), dtype='uint8')     # 3 channels
    new_mask = np.zeros((mo_h, mo_w),dtype='float')

    # split points
    div_point_x = int(mo_w * scale_x)
    div_point_y = int(mo_h * scale_y)

    # loop through images
    k =random.randint(0,3) # choose which part to be as foreground
    for i in range(len(imgs)):
        # top left image, img_0
        
        if i == 0:
            # width and height of the top left image
            w0 = div_point_x
            h0 = div_point_y
            img_0,mask_0,points_0, bboxes_0, class_labels_0 = random_crop_savebboxes(imgs[0], masks[0],all_points[0], bboxes[0] , labels[0] ,h0 , w0, min_area, min_visibility)
            # top left
            new_img[:div_point_y, :div_point_x, :] = img_0
            if i ==k:
                new_mask[:div_point_y, :div_point_x] = mask_0

                # change parameters of bboxes for the top left image, chú ý bboxes ở dạng list of tuple mà tuple ko cho gán nên ta tạo list mới
                if len(bboxes_0) == 0:  # there is no boxes
                    bboxes_0_new = []
                    points_0_new = []
                elif len(points_0) == 0:  # there is no boxes
                    bboxes_0_new = []
                    points_0_new = []
                else:
                    # !!! Don't create nested list as follow: bboxes_0_new = [[None, None, None, None]] * len(bboxes_0)
                    # because inner lists have the same id => incorrect result when perform operations.
                    
                    # nested list for bboxes
                    bboxes_0_new = np.zeros((len(bboxes_0), 4))
                    points_0_new = np.zeros((len(points_0),2))
                    # convert to list
                    bboxes_0_new = bboxes_0_new.tolist()
                    points_0_new = points_0_new.tolist()

                    for i, box in enumerate(bboxes_0):
                        for x in box:
                            if x < 0.0:
                                x =0.0005
                            if x >1:
                                x = 0.9998
                        bboxes_0_new[i][0] = box[0] * scale_x   # xcenter
                        bboxes_0_new[i][2] = box[2] * scale_x   # w

                        bboxes_0_new[i][1] = box[1] * scale_y   # ycenter
                        bboxes_0_new[i][3] = box[3] * scale_y   # h
                    for i, pt in enumerate(points_0):
                        points_0_new[i][0] = pt[0]    # x_center
                        points_0_new[i][1] = pt[1]    # y_center
                
                new_bboxes = [bboxes_0_new]
                new_points = [points_0_new]
        # top right image
        elif i == 1:
            
            w1 = mo_w - div_point_x     # trừ sẽ khớp
            h1 = div_point_y    # giữ nguyên như cái i=0
            img_1, mask_1,points_1, bboxes_1, class_labels_1 = random_crop_savebboxes(imgs[1], masks[1], all_points[1], bboxes[1], labels[1], h1, w1, min_area, min_visibility)
            new_img[:div_point_y, div_point_x:, :] = img_1
            if i == k:
                new_mask[:div_point_y, div_point_x:] = mask_1

                # change parameters of bboxes for the top left image, chú ý bboxes ở dạng list of tuple mà tuple ko cho gán nên ta tạo list mới
                if len(bboxes_1) == 0:  # there is no boxes
                    bboxes_1_new = []
                    points_1_new = []
                elif len(points_1) == 0:
                    bboxes_1_new = []
                    points_1_new = []
                else:
                    # !!! Don't create nested list as follow: bboxes_0_new = [[None, None, None, None]] * len(bboxes_0)
                    # because inner lists have the same id => incorrect result when perform operations.
                    
                    # nested list for bboxes
                    bboxes_1_new = np.zeros((len(bboxes_1), 4))
                    points_1_new = np.zeros((len(points_1),2))
                    # convert to list
                    bboxes_1_new = bboxes_1_new.tolist()
                    points_1_new = points_1_new.tolist()

                    for i, box in enumerate(bboxes_1):
                        for x in box:
                            if x < 0.0:
                                x =0.0005
                            if x >1:
                                x = 0.9998
                        bboxes_1_new[i][0] = box[0] * (1 - scale_x) + scale_x   # xcenter
                        bboxes_1_new[i][2] = box[2] * (1 - scale_x)   # w

                        bboxes_1_new[i][1] = box[1] * scale_y   # ycenter
                        bboxes_1_new[i][3] = box[3] * scale_y   # h
                    for i, pt in enumerate(points_1):
                        points_1_new[i][0] = pt[0] + div_point_x    # x_center
                        points_1_new[i][1] = pt[1]   # y_center
                    
                new_bboxes = [bboxes_1_new]
                new_points = [points_1_new]
        # bottom left image
        elif i == 2:
            
            w2 = div_point_x
            h2 = mo_h - div_point_y
            img_2, mask_2,points_2, bboxes_2, class_labels_2 = random_crop_savebboxes(imgs[2], masks[2], all_points[2], bboxes[2], labels[2], h2, w2, min_area, min_visibility)
            new_img[div_point_y:, :div_point_x, :] = img_2
            if i==k:
                new_mask[div_point_y:, :div_point_x] = mask_2

                # change parameters of bboxes for the top left image, chú ý bboxes ở dạng list of tuple mà tuple ko cho gán nên ta tạo list mới
                if len(bboxes_2) == 0:  # there is no boxes
                    bboxes_2_new = []
                    points_2_new = []
                elif len(points_2) == 0:  # there is no boxes
                    bboxes_2_new = []
                    points_2_new = []
                else:
                    # !!! Don't create nested list as follow: bboxes_0_new = [[None, None, None, None]] * len(bboxes_0)
                    # because inner lists have the same id => incorrect result when perform operations.
                    
                    # nested list for bboxes
                    bboxes_2_new = np.zeros((len(bboxes_2), 4))
                    points_2_new = np.zeros((len(points_2),2))
                    # convert to list
                    bboxes_2_new = bboxes_2_new.tolist()
                    points_2_new = points_2_new.tolist()

                    for i, box in enumerate(bboxes_2):
                        for x in box:
                            if x < 0.0:
                                x =0.0005
                            if x >1:
                                x = 0.9998
                        bboxes_2_new[i][0] = box[0] * scale_x   # xcenter
                        bboxes_2_new[i][2] = box[2] * scale_x   # w

                        bboxes_2_new[i][1] = box[1] * (1 - scale_y) + scale_y       # ycenter
                        bboxes_2_new[i][3] = box[3] * (1 - scale_y)                 # h
                    for i, pt in enumerate(points_2):
                        points_2_new[i][0] = pt[0]    # x_center
                        points_2_new[i][1] = pt[1] + div_point_y    # y_center
                
                new_bboxes = [bboxes_2_new]
                new_points = [points_2_new]
        # bottom right image
        else:
            
            w3 = mo_w - div_point_x
            h3 = mo_h - div_point_y
            img_3, mask_3, points_3, bboxes_3, class_labels_3 = random_crop_savebboxes(imgs[3], masks[3], all_points[3], bboxes[3], labels[3], h3, w3, min_area, min_visibility)
            new_img[div_point_y:, div_point_x:, :] = img_3
            if i == k:
                new_mask[div_point_y:, div_point_x:] = mask_3 

                # change parameters of bboxes for the top left image, chú ý bboxes ở dạng list of tuple mà tuple ko cho gán nên ta tạo list mới
                if len(bboxes_3) == 0:  # there is no boxes
                    bboxes_3_new = []
                    points_3_new = []
                elif len(points_3) == 0:  # there is no boxes
                    bboxes_3_new = []
                    points_3_new = []
                else:
                    # !!! Don't create nested list as follow: bboxes_0_new = [[None, None, None, None]] * len(bboxes_0)
                    # because inner lists have the same id => incorrect result when perform operations.
                    
                    # nested list for bboxes
                    bboxes_3_new = np.zeros((len(bboxes_3), 4))
                    points_3_new = np.zeros((len(points_3),2))
                    # convert to list
                    bboxes_3_new = bboxes_3_new.tolist()
                    points_3_new = points_3_new.tolist()

                    for i, box in enumerate(bboxes_3):
                        for x in box:
                            if x < 0.0:
                                x =0.0005
                            if x >1:
                                x = 0.9998
                        bboxes_3_new[i][0] = box[0] * (1 - scale_x) + scale_x   # xcenter
                        bboxes_3_new[i][2] = box[2] * scale_x   # w

                        bboxes_3_new[i][1] = box[1] * (1 - scale_y) + scale_y   # ycenter
                        bboxes_3_new[i][3] = box[3] * scale_y   # h
                    for i, pt in enumerate(points_3):
                        points_3_new[i][0] = pt[0] + div_point_x   # x_center
                        points_3_new[i][1] = pt[1] + div_point_y    # y_center
                
                new_bboxes = [bboxes_3_new]
                new_points = [points_3_new]
    # all classes in the augmented image
    # new_class_labels = [class_labels_0 , class_labels_1 , class_labels_2 , class_labels_3]
    # all bounding boxex in the augmented image

    #new_bboxes = [bboxes_0_new , bboxes_1_new , bboxes_2_new , bboxes_3_new]
    return new_img, new_mask, new_bboxes, new_points

def get_individual_augmentations():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.3),
        A.Blur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.2),
    ])