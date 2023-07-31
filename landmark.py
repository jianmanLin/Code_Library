import face_alignment
import numpy as np
import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from skimage import io
from scipy.ndimage import gaussian_filter1d

### 输入图像获取landmarks
def get_landmark(img):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    lm = []
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    preds = detector.get_landmarks(img) 
    if len(preds) == 0:  # 没有检测到面部
        return None
    for kk in range(68):
        lm.append((preds[0][kk][0], preds[0][kk][1]))
    lm = np.array(lm)
    return lm

### 获取整个目录下图片的landmarks
def video2sequence_landmark(param):
    ## 保存路径
    save_path = os.path.join("/data2/JM/code/DiffTalk-main/HDTF/landmark256_aligned", param[1])
    image_folder = os.path.join(param[0], param[1])
    os.makedirs(save_path, exist_ok=True)
    # landmarkpredictor = dlib.shape_predictor('/data2/JM/code/DiffTalk-main/models/shape_predictor_68_face_landmarks.dat')
    for img_name in sorted(os.listdir(image_folder)):
        img_p = os.path.join(image_folder, img_name)
        img = cv2.imread(img_p)
        lm = get_landmark(img)
        if lm is not None:
            np.savetxt(f'{save_path}/{img_name[:-4]}.txt', lm, fmt='%d', delimiter=',', encoding='utf-8')



### 将landmarks贴回人脸
def plt_landmarks(img, landmarks):
    # 在原始图像上绘制标记点
    for kk in range(68):
        img_lm = cv2.circle(img, (int(landmarks[kk][0]),int(landmarks[kk][1])), radius=3, color=(255, 0, 255), thickness=-1)
    # 显示图像
    cv2.imwrite('Landmarked.png', img)

## 输入格式：
## 主目录：
##   子目录1
##   子目录2
##   每个子目录为一个视频的图片--多线程提取landmarks，以txt的格式保存
def main():
    params = []
    root_path = '/data2/JM/code/DiffTalk-main/HDTF/img_256_aligned'
    root_list = os.listdir(root_path)
    for folder in root_list:
        params.append([root_path, folder])
    p = Pool(2)
    for _ in tqdm(p.imap_unordered(video2sequence_landmark, params), total=len(params)):
        pass


## 进行人脸对齐--光流追获取landmark--将landmarks贴会原脸检测--对齐人脸
def align_frames(img_dir, save_dir, output_size=1024, transform_size=1024, optical_flow=True, gaussian=False, filter_size=3):
    os.makedirs(save_dir, exist_ok=True)
    # load face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    # list images in the directory
    img_list = os.listdir(img_dir)
    img_list.sort()
    # save align statistics
    stat_dict = {'quad':[], 'qsize':[], 'coord':[], 'crop':[]}
    lms = []
    from tqdm import tqdm
    for idx, img_name in tqdm(enumerate(img_list)):
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        lm = []
        ## [1, 68, 2]
        preds = fa.get_landmarks(img) 
        ## [68, 2]
        for kk in range(68):
            lm.append((preds[0][kk][0], preds[0][kk][1]))
        # Eye distance
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_to_eye   = eye_right - eye_left
        if optical_flow:
            if idx > 0:
                ## 用于计算给定两个数组中相应元素的平方和的平方根
                s = int(np.hypot(*eye_to_eye)/4)
                ## 光流跟踪
                lk_params = dict(winSize=(s, s), maxLevel=5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))
                ## 结构处理
                points_arr = np.array(lm, np.float32)
                points_prevarr = np.array(prev_lm, np.float32)
                ## 兴趣点在下一帧图像中的位置 || 光流跟踪是否成功的标识 || 试的错误数组
                points_arr,status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, points_prevarr, points_arr, **lk_params)
                sigma =100
                points_arr_float = np.array(points_arr,np.float32)
                points = points_arr_float.tolist()
                for k in range(0, len(lm)):
                    d = cv2.norm(np.array(prev_lm[k]) - np.array(lm[k]))
                    alpha = np.exp(-d*d/sigma)
                    lm[k] = (1 - alpha) * np.array(lm[k]) + alpha * np.array(points[k])
            prev_img = img
            prev_lm = lm
        lms.append(lm)
    # Apply gaussian filter on landmarks
    if gaussian:
        lm_filtered = np.array(lms)
        for kk in range(68):
            lm_filtered[:, kk, 0] = gaussian_filter1d(lm_filtered[:, kk, 0], filter_size)
            lm_filtered[:, kk, 1] = gaussian_filter1d(lm_filtered[:, kk, 1], filter_size)
        lms = lm_filtered.tolist()
    # save landmarks
    landmark_out_dir = os.path.dirname(img_dir) + '_landmark/'
    os.makedirs(landmark_out_dir, exist_ok=True)

    for idx, img_name in tqdm(enumerate(img_list)):
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        lm = lms[idx]
        img_lm = img.copy()
        for kk in range(68):
            img_lm = cv2.circle(img_lm, (int(lm[kk][0]),int(lm[kk][1])), radius=3, color=(255, 0, 255), thickness=-1)
        # Save landmark images
        cv2.imwrite(landmark_out_dir + img_name, img_lm[:,:,::-1])
        # Save mask images
        """
        seg_mask = np.zeros(img.shape, img.dtype)
        poly = np.array(lm[0:17] + lm[17:27][::-1], np.int32)
        cv2.fillPoly(seg_mask, [poly], (255, 255, 255))
        cv2.imwrite(img_dir + "mask%04d.jpg"%idx, seg_mask);
        """
        # Parse landmarks. 
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        # Calculate auxiliary vectors.
        eye_left     = np.mean([lm_eye_left[0], lm_eye_left[3]], axis=0)
        eye_right    = np.mean([lm_eye_right[0], lm_eye_right[3]], axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = np.array(lm_mouth_outer[0])
        mouth_right  = np.array(lm_mouth_outer[6])
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        stat_dict['coord'].append(quad)
        stat_dict['qsize'].append(qsize)

    # Apply gaussian filter on crops
    if gaussian:
        quads = np.array(stat_dict['coord'])
        quads = gaussian_filter1d(quads, 2*filter_size, axis=0)
        stat_dict['coord'] = quads.tolist()
        qsize = np.array(stat_dict['qsize'])
        qsize = gaussian_filter1d(qsize, 2*filter_size, axis=0)
        stat_dict['qsize'] = qsize.tolist()

    for idx, img_name in tqdm(enumerate(img_list)):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        qsize = stat_dict['qsize'][idx]
        quad = np.array(stat_dict['coord'][idx])
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
        stat_dict['crop'].append(crop)
        stat_dict['quad'].append((quad + 0.5).flatten())
        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        # resizing
        img_pil = img.resize((output_size, output_size), Image.LANCZOS)
        img_pil.save(save_dir+"/"+img_name)
    # create_video(landmark_out_dir)
    np.save(save_dir+'stat_dict.npy', stat_dict)


## 获取嘴部区域mask
def get_mouth_mask(landmarks, image):
    h, w = image.size
    #inference mask  -- mask掉嘴部区域
    inference_mask = np.ones((h, w))
    points = landmarks[2:15].astype('int32')
    # points = np.concatenate((points, landmarks[33:34])).astype('int32')
    inference_mask = cv2.fillPoly(inference_mask, [points], (0, 0, 0), 8, 0)
    # inference_mask = cv2.fillPoly(inference_mask, [points], (0), 8, 0)
    inference_mask = (inference_mask > 0).astype(int)
    inference_mask = Image.fromarray(inference_mask.astype(np.uint8))
    inference_mask = inference_mask.resize((64, 64), resample="bicubic")
    inference_mask = np.array(inference_mask)

## 只获取人脸的上部分landmarks，不包含嘴部区域
def get_uplandmarks(example):
    landmarks = np.loadtxt(example["landmark_path_"], delimiter=',', dtype=np.float32)
    landmarks_img = landmarks[13:48]
    landmarks_img2 = landmarks[0:4]
    landmarks_img = np.concatenate((landmarks_img2, landmarks_img))

if __name__ == "__main__":
    test_img = "/data2/JM/code/code_library/test_image/img1.jpg"
    align_frames("/data2/JM/code/code_library/test_img/img", "/data2/JM/code/code_library/test_img/img_aligned")