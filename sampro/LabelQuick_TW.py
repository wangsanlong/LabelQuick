import numpy as np
import cv2
from sampro.sam2.build_sam import build_sam2
from sampro.sam2.sam2_image_predictor import SAM2ImagePredictor
import sys

#设置当前文件夹
sys.path.append(r'sampro')

class Anything_TW():
    def __init__(self):
        self.sam2_checkpoint = "sampro\checkpoints\sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = "cuda"
        #全局变量
        self.coords = []
        self.methods = []
        
        self.click_open = False
        self.option = False
        
        self.clicked_x = None
        self.clicked_y = None
        self.method = None
        
        #预测的结果
        self.masks = None
        self.scores = None
        self.logits = None
        
        self.mask = None
        
        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    #设置图像
    def Set_Image(self, image):
        self.predictor.set_image(image)
                
        #初始图像
        self.image = image.copy()
        #画了点的图像
        self.image_dot = image.copy()  
        #画了mask的图像
        self.image_mask = image.copy()
        #用来保存mask的图像
        self.image_save = image.copy()
        
        
    #设置点击 
    def Set_Clicked(self, clicked, method):
        self.clicked_x, self.clicked_y = clicked
        self.method = method
        
    #键盘点击事件
    def Key_Event(self, key):
        if key == 83:
            self.image_save = self.Draw_Mask(self.mask, self.image_save)
            
            self.image_dot = self.image.copy()
            self.image_mask = self.image_save.copy()
            
            self.coords = []
            self.methods = []
            
            
        elif key == 81:
            self.image_dot = self.image.copy()
            self.image_mask = self.image_save.copy()
            
            self.coords = []
            self.methods = []
            
        #键盘的backspace键
        elif key == 16777219:

            self.image_dot = self.image.copy()
            self.image_mask = self.image.copy()
            self.image_save = self.image.copy()
                        
            self.coords = []
            self.methods = []
            
        return self.image_mask
            
            
    
    def _pick_best_mask_containing_point(self, masks, px, py):
        """
        从多个 mask 中选“包含点击点且面积最小”的，保证点击哪头牛就选哪头。
        若都不包含点击点，则选 bbox 中心离点击点最近的。
        """
        best_mask = None
        best_area = float('inf')
        fallback_mask = None
        fallback_dist = float('inf')
        for m in masks:
            h, w = m.shape[-2:]
            m_uint8 = (m.reshape(h, w).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0:
                    area = cv2.contourArea(c)
                    if area > 10 and area < best_area:
                        best_area = area
                        best_mask = m
                    break
            if best_mask is None:
                all_pts = np.vstack(contours) if contours else np.array([[0, 0], [1, 1]])
                x, y, bw, bh = cv2.boundingRect(all_pts)
                cx, cy = x + bw / 2, y + bh / 2
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < fallback_dist:
                    fallback_dist = d
                    fallback_mask = m
        return best_mask if best_mask is not None else (fallback_mask if fallback_mask is not None else masks[-1])

    #显示点
    def Draw_Point(self, image,label):
        if label == 1:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (255, 0, 0), -1) 
        elif label == 0:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (0, 0, 255), -1)
            
    #创建Mask
    def Create_Mask(self):
        self.coords.append([self.clicked_x, self.clicked_y])
        self.methods.append(self.method)
        
        if self.option == False:
            input_point = np.array(self.coords)
            input_method = np.array(self.methods)

            self.masks, self.scores, self.logits = self.predictor.predict(
                point_coords = input_point,
                point_labels = input_method,
                multimask_output = True,
            )
            self.option = True
            # 从多个 mask 中选“包含点击点且面积最小”的，保证点击哪头牛就选哪头
            self.mask = self._pick_best_mask_containing_point(
                self.masks, self.clicked_x, self.clicked_y
            )

        else:
            input_point = np.array(self.coords)
            input_method = np.array(self.methods)
            mask_input = self.logits[np.argmax(self.scores), :, :]  # Choose the model's best mask

            self.masks, self.scores, self.logits  = self.predictor.predict(
                point_coords = input_point,
                point_labels = input_method,
                mask_input = mask_input[None, :, :],
                multimask_output = False,
            )
            #  refinement 后：左键选包含点击点的，右键选最大（因点击处已被排除）
            if self.method == 1:
                self.mask = self._pick_best_mask_containing_point(
                    self.masks, self.clicked_x, self.clicked_y
                )
            else:
                self.mask = self.masks[-1] 
    
    #画Mask
    def Draw_Mask(self, mask, image):
        # 获取轮廓
        h,w = mask.shape[-2:]
        mask = mask.reshape(h,w,1)
        mask = mask.astype(np.uint8)

        white = np.zeros([h,w,1],dtype="uint8")
        white[:,:,0] = 255
        x = mask * white
        canny = cv2.Canny(x,50,100)
        # 画一个实心圆
        self.Draw_Point(image,self.method)
        img = image.copy()
        # 找到边缘轮廓
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 优先选择“包含点击点”的轮廓，保证框的中心对象就是你点击的对象
        selected_contour = None
        candidate_area = 0
        for contour in contours:
            # 先判断点击点是否在该轮廓内部（>=0 表示在轮廓上或内部）
            if cv2.pointPolygonTest(contour, (float(self.clicked_x), float(self.clicked_y)), False) >= 0:
                area = cv2.contourArea(contour)
                # 在所有包含点击点的轮廓中，选择面积最大的一个
                if area > candidate_area:
                    candidate_area = area
                    selected_contour = contour

        # 如果没有任何轮廓包含点击点，则退回到“选择最大面积轮廓”的旧策略
        if selected_contour is None:
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    selected_contour = contour

        # 使用选中的轮廓绘制矩形框
        self.x, self.y, self.w, self.h = cv2.boundingRect(selected_contour)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        # 在原图上绘制边缘线
        cv2.drawContours(img, [selected_contour], -1, (0, 255, 0), 2)
        self.image_mask = img
        return img
    
    
if __name__ == '__main__':
    
    image = cv2.imread(r'segment\notebooks\images\1.png')
    img = cv2.resize(image, (1000, 800))
    AD = Anything_TW()
    AD.Set_Image(img)
    AD.Set_Clicked([200,500],1)

    AD.Create_Mask()
    # AD.Key_Event("s")
    AD.Draw_Mask(AD.mask, AD.image_mask)
    cv2.imshow('Image1', AD.image_mask)
    cv2.waitKey(0)