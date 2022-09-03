import os
import cv2 
import json
import math
import time   
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.spatial import distance as dist

class OpticalTrack():
    previewWindow = True
    
    # params for Shi-Tomasi corner detection
    shitomasi_params = {"qualityLevel": 0.1,"minDistance": 7,"blockSize": 7}

    # params for Lucas-Kanade optical flow
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    LK_params = {"winSize": (15,15),"maxLevel": 2,"criteria": criteria}

    # params for visualization
    color = np.random.randint(0,255,(100,3))
    trailThickness = 8  # thickness of the trail to draw behind the target
    trailFade = 4       # the intensity at which the trail fades
    pointSize = 15      # pixel radius of the circle to draw over tracked points


    def __init__(self,numPts=30,trailLength=50,savevid=False) -> None:
        self.numPts = numPts            # max number of points to track
        self.trailLength = trailLength  # numeber of past time frames to keep 
        self.savevid = savevid          # whether to save result video

        self.orients_previous = np.zeros((1,2))     # overall oriention vector of the previous time
        self.orient_history = [0 for i in range(25)]# orientation history of past frames
        self.trail_states = np.zeros((numPts,2))    # whether a point is still being tracked
        self.trail_history = [[[(0,0), (0,0)] for j in range(trailLength)] for i in range(numPts)]
        
      
    def opticalTrack(self,camera_id,roi):
        start_time = time.time()

        # SETUP -----------------------------------------------------------------------
        # read the video file into memory
        cap = cv2.VideoCapture(camera_id)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(desc='FPS:0',total=num_frames)
        count = 1
        print("FPS: ",fps)

        # get the first frame
        _, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # create crosshair mask 
        crosshairmask = np.zeros_like(old_gray)
        crosshairmask = cv2.fillConvexPoly(crosshairmask,roi,color=255)
        visualmask = cv2.fillConvexPoly(np.zeros_like(old_frame),roi,color=(255,255,255))

        # create roi mask
        mask = cv2.fillConvexPoly(np.zeros_like(old_gray),roi,color=255)

        # # get trail and orient info
        # orients_previous = self.orients_previous
        # orient_history = self.orient_history
        # trail_states = self.trail_states
        # trail_history = self.trail_history

        # get features from first frame
        print(f"\nRunning Optical Flow on: {camera_id}")
        old_points = cv2.goodFeaturesToTrack(old_gray, maxCorners=self.numPts, mask=crosshairmask, **self.shitomasi_params)
        points_id = np.arange(len(old_points),dtype=np.int32).reshape(len(old_points),1)  # feature point id in trail_history

        # if saving video
        if self.savevid:
            # path to save output video
            savepath = camera_id.split('.')[-2] + '_LK_FLOW' + '.mp4'
            print(f"Saving Output video to: {savepath}")

            # get shape of video frames
            height, width, _ = old_frame.shape

            # setup videowriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoOut = cv2.VideoWriter(savepath, fourcc, fps, (width, height))

        # PROCESS VIDEO ---------------------------------------------------------------
        while(True):
            start = time.time()
            # get next frame and convert to grayscale
            stillGoing, new_frame = cap.read()

            # if video is over, quit
            if not stillGoing:
                break
            else:
                count += 1

            # convert to grayscale
            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **self.LK_params)

            # select good points
            if old_points is not None:
                good_new = new_points[st==1]
                good_old = old_points[st==1]
                good_id = points_id[st==1]

            # create trail mask to add to image
            trailMask = np.zeros_like(old_frame)

            # calculate motion lines and points, and check points according to history trail
            valid_ids = []
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                # flatten coords
                a,b = new.ravel()
                c,d = old.ravel()
                k = good_id[i]

                # list of the prev and current points converted to int
                linepts = [(int(a),int(b)), (int(c),int(d))]

                # add points to the trail history
                self.trail_history[k].insert(0, linepts)

                # check valid of trail
                isvalid = self.check_flow(self.trail_history[k])
                if isvalid:
                    valid_ids.append(i)
                    # points update times
                    self.trail_states[k,1] += 1

                # add trail lines to image
                if self.previewWindow and isvalid:
                    self.draw_trail(trailMask,self.trail_history,k,new_frame)
                
                # get rid of the trail segment
                self.trail_history[k].pop()

            # set trail states, 0 for missed, 1 for still being tracked
            self.trail_states[:, 0] = 0
            self.trail_states[good_id[valid_ids], 0] = 1

            # filter invalid flow
            self.trail_history = self.filter_invalid_flow(self.trail_history,lateral_thresh=50)

            # Calculation direction
            orient = self.get_orientation(self.trail_history,self.trail_states,self.orients_previous,count)
            orient = self.update_orient(self.orient_history,orient)
            if orient==1:
                cv2.putText(new_frame,'down',(20,50),cv2.LINE_AA,2,(255,0,0),2)
            elif orient==-1:
                cv2.putText(new_frame,'up',(20,50),cv2.LINE_AA,2,(255,0,0),2)
            else:
                cv2.putText(new_frame,'-',(20,50),cv2.LINE_AA,2,(255,0,0),2)
            
            # add trail to frame
            new_frame = cv2.addWeighted(new_frame,0.7,visualmask,0.3,0)
            img = cv2.add(new_frame, trailMask)

            # show the frames
            if self.previewWindow:
                cv2.imshow('optical flow', img)

            # write frames to new output video
            if self.savevid:
                videoOut.write(img)

            # kill window if ESC is pressed
            rest = max(1/fps - (time.time()-start),1)
            k = cv2.waitKey(rest) & 0xff
            if k == 27:
                break
            
            # update previous frame and previous points
            old_gray = new_frame_gray.copy()
            old_points = good_new[valid_ids,...].reshape(-1,1,2)
            points_id = good_id[valid_ids,...].reshape(-1,1)
            
            # if old_points < numPts, get new points
            old_points,points_id = self.update_points(old_gray,old_points,points_id,mask,crosshairmask)
            # trail_states = self.trail_states
            # trail_history = self.trail_history
            
            FPS = 1/(time.time()-start)
            AVG_FPS = count/(time.time()-start_time)
            pbar.set_description_str('FPS:{:.0f} AVG_FPS:{:.2f} orient:{}'.format(FPS,AVG_FPS,orient))
            pbar.update()
            assert len(points_id) == len(old_points)
        # after video is finished
        pbar.close()
        print('\nComplete!\n')
    
    def draw_trail(self,trailMask,trail,k,frame):
        fade = self.trailFade
        color = self.color[k].tolist() # get color for this point
        thickness = self.trailThickness
        pointSize = self.pointSize
        # draw trail
        for j in range(len(trail[k])):
            pt1,pt2 = tuple(trail[k][j][0]),tuple(trail[k][j][1])
            trailColor = [int( color[0]-(fade*j) ), int( color[1]-(fade*j) ), int( color[2]-(fade*j) )] # fading colors
            trailMask = cv2.line(trailMask, pt1, pt2, trailColor, thickness=thickness, lineType=cv2.LINE_AA)
        # add circle over the point
        cv2.circle(frame, tuple(trail[k][0][0]), pointSize, color, -1)

    def filter_invalid_flow(self,multi_pts,lateral_thresh=30):
        '''filter flow with large lateral displacement\n
        Arguments:
            multi_points: trail history
            lateral_thresh: threshold of lateral diplacement, pixel distance
        Return:
            filterd multi_points 
        ''' 
        multi_pts = np.array(multi_pts)
        zeros_pts = np.zeros_like(multi_pts)
        displacement = multi_pts[:,:,0,:]-multi_pts[:,:,1,:]
        multi_pts = np.where(displacement[...,0][...,None,None]>lateral_thresh,zeros_pts,multi_pts).astype(np.int32)
        return multi_pts.tolist()

    def get_orientation(self,trail_history,states,orients_previous=None,penalty=5,update_weight=0.7,stop_thresh=10):
        '''calculate orientation according to trail history\n
        Arguments:
            trail_history: history coordinates of feature points, shape: numPts x trailLength x 2 x 2
            states: states of feature points, wheather it has been tracking and update times, shape: numPts x 2
            orients_previsous: previous overall direction of escalator, shape: 1 x 2
            penalty: attenuated weight of inactive points trail history when calculate direction
            update_weigth: weight of new dispalcement, orient = update_weigth*orients + (1-update_weigths)*orients_previous 
            stop_thresh: when the overall dispalcement is lower than this threshold, the direction will be judged as stop
        Return:
            orient: current direction of escalator
        '''
        
        multi_pts = np.array(trail_history)
        active_ids = states[:,0]==1
        active_updates = states[active_ids,1]
        active_pts = multi_pts[active_ids,...]
        inactive_ids = states[:,0]==0
        inactive_updates = states[inactive_ids,1]
        inactive_pts = multi_pts[inactive_ids,...]
        for i,state in enumerate(inactive_ids):
            if state:
                # attenuate the displacement of feature points that are not continuously tracked
                trail_history[i]=(np.array(trail_history[i])/10).astype(np.int32).tolist() 
        
        active_disp = active_pts[:,:,0,:]-active_pts[:,:,1,:]
        active_nums = np.sum(active_updates)
        active_orients = np.sum(active_disp,axis=(0,1)) if active_nums!=0 else np.zeros(2,dtype=np.int32)
        
        inactive_disp = inactive_pts[:,:,0,:]-inactive_pts[:,:,1,:]
        inactive_nums = np.sum(inactive_updates)
        inactive_orients = np.sum(inactive_disp,axis=(0,1))/penalty if inactive_nums!=0 else np.zeros(2,dtype=np.int32)
        
        orients_previous = np.zeros((1,2)) if orients_previous is None else orients_previous
        orients = update_weight*(active_orients+inactive_orients)[None,:] + orients_previous*(1-update_weight)

        angle = np.arctan2(orients[:,1],orients[:,0])/np.pi*180
        angle = np.nanmean(angle)

        orients_previous[0,0] = orients[0,0]
        orients_previous[0,1] = orients[0,1]
        
        if -150<angle<-30 and orients[0,1]<-stop_thresh:
            orient = -1 # ↑
        elif 30<angle<150 and orients[0,1]>stop_thresh:
            orient = 1  # ↓
        else:
            orient = 0  # -
        return orient

    def get_default_pts(self,roi,num_pts=60):
        '''get random feature points when opencv function can't get good feature\n
        Arguments:
            roi: roi area of escalator
            num_pts: number of feature points needed
        Return:
            pts: defaulf feature points in roi area
        '''
        left = np.min(roi[:,0])
        right = np.max(roi[:,0])
        top = np.min(roi[:,1])
        down = np.max(roi[:,1])
        pts_x = np.random.randint(left,right,num_pts)
        pts_y = np.random.randint(top,down,num_pts)
        pts = np.stack([pts_x,pts_y],axis=-1)
        pts = pts[:,None,:].astype(np.float32)
        return pts

    def get_FeaturesToTrack_deprecated(self,image,maxCorners,mask,roi):
        qualityLevel = self.shitomasi_params["qualityLevel"]
        minDistance = self.shitomasi_params["minDistance"]
        blockSize = self.shitomasi_params["blockSize"]
        feat_pts = None
        try:
            qualityLevel_loose = max(qualityLevel-0.1,0.1)
            feat_pts = cv2.goodFeaturesToTrack(image,maxCorners,qualityLevel_loose,minDistance,mask=mask,blockSize=blockSize)
            assert feat_pts is not None,'\nNone features points have been tracked, try to loose quality level'
        except:
            qualityLevel_loose = max(qualityLevel-0.2,0.1)
            feat_pts = cv2.goodFeaturesToTrack(image,maxCorners,qualityLevel_loose,minDistance,mask=mask,blockSize=blockSize)
            print(f'\nquality level has been loosed from {qualityLevel} to {qualityLevel_loose:.1f}')
        finally:
            if feat_pts is None:
                feat_pts = self.get_default_pts(roi,maxCorners)
                print('\nNone features points have been tracked, set default features points')
                
        return feat_pts

    def get_FeaturesToTrack(self,image,maxCorners,mask,roi):
        '''get new feature points\n
        Arguments:
            image: gray image of in which feature points to get
            maxCorners: number of feature points needed
            mask: roi mask
            roi: roi area
        Return:
            feat_pts: feature points to be tracked
        '''
        qualityLevel = self.shitomasi_params["qualityLevel"]
        minDistance = self.shitomasi_params["minDistance"]
        blockSize = self.shitomasi_params["blockSize"]
        feat_pts = None
        try:
            feat_pts = cv2.goodFeaturesToTrack(image,maxCorners,qualityLevel,minDistance,mask=mask,blockSize=blockSize)
        finally:
            if feat_pts is None:
                feat_pts = self.get_default_pts(roi,maxCorners)
                print('\nNone features points have been tracked, set default features points')
        return feat_pts

    def check_flow(self,flow):
        '''filter points whose direction are changing strangely, somtimes up, somtimes down.
        '''
        flow = np.array(flow)
        flow = flow[::5,...]
        disp = flow[:-1,0,1]-flow[1:,0,1]
        cur_orient = 0
        reverse_times = 0
        for i in range(len(disp)):
            if disp[i]>5:
                orient = 1
            elif disp[i]<-5:
                orient = -1
            else:
                orient = 0
            if abs(cur_orient-orient)>1:
                reverse_times+=1
            cur_orient = orient if orient!=0 else cur_orient
        return False if reverse_times>2 else True

    def update_orient(self,orient_history,orient):
        '''update orient according to the orient appeared most times in history orient
        '''
        orient_history.insert(0,orient)
        orient_history.pop()
        orient = Counter(orient_history).most_common(1)[0][0]
        return orient

    def update_points(self,frame_gray,points,points_id,roi_mask,crosshairmask):
        '''determine whether the feature points need to be updated
        '''
        numPts = self.numPts
        trailLength = self.trailLength
        
        # filter feature points out of ROI 
        coord_x = np.asarray(points[:,0,0], dtype=np.int64)
        coord_y = np.array(points[:,0,1], dtype=np.int64)
        valid_indices = np.where(roi_mask[coord_y,coord_x]==255,True,False)
        points = points[valid_indices]
        points_id = points_id[valid_indices]

        # if old_points < numPts, get new points
        if (numPts - len(points)) > 10 :
            # reset trail staet and trail history
            self.trail_states = np.zeros((numPts,2))
            self.trail_history = [[[(0,0), (0,0)] for j in range(trailLength)] for i in range(numPts)]
            # update points and points id
            points = self.get_FeaturesToTrack(frame_gray, maxCorners=numPts, mask=crosshairmask, roi=roi)
            points_id = np.arange(len(points),dtype=np.int32).reshape(len(points),1)

        return points,points_id

class ROI_GETER():
    '''ROI for direction judge
    '''
    class Line:
        def __init__(self,pt1,pt2,id=0) -> None:
            self.pt1 = pt1
            self.pt2 = pt2
            self.id = id
            self.x_low = min(pt1[0],pt2[0])
            self.x_up  = max(pt1[0],pt2[0])
            self.y_low = min(pt1[1],pt2[1])
            self.y_up  = max(pt1[1],pt2[1])
            self.k, self.b = self._get_kb(pt1,pt2)
        
        def _get_kb(self,pt1,pt2):
            # 获取线段的斜率和截距
            x1,y1 = pt1
            x2,y2 = pt2
            k = None if x1 == x2 else (y1-y2)/(x1-x2)
            b = None if k is None else y1-k*x1
            return k,b

        def is_inrange(self,x=None,y=None):
            x_inrange = True
            y_inrange = True
            if x is not None:
                x_inrange = self.x_low <= x <= self.x_up
            if y is not None:
                y_inrange = self.y_low <= y <= self.y_up
            return x_inrange and y_inrange
  
        def __call__(self, x=None,y=None, *args, **kwds):
            '''输入x坐标, 则计算对应的y坐标; 输入y坐标, 则计算对应的x坐标\n
            '''
            assert (x is None) ^ (y is None ),'x and y should not be specified or None at the same time'
            result = 0
            if x is not None:  
                if self.k is None:
                    result = None       # 此时线段与y轴平行，给定x坐标无法求出y坐标
                else:
                    result = self.k*x + self.b
            if y is not None:
                if self.k is None:
                    result = self.x_low # 此时线段与y轴平行，x坐标为固定值
                elif self.k == 0:
                    result = None       # 此时线段与y轴平行，给定y坐标无法求出x坐标
                else:
                    result = (y-self.b)/self.k
            return result
        

    def __init__(self,contour,img_shape) -> None:
        self.contour = contour
        self.lines = self._parse_contour(contour)
        self.img_shape = img_shape

    def _parse_contour(self,contour):
        lines = []
        for i,(pt1,pt2) in enumerate(zip(contour[:-1,:],contour[1:,:])):
            lines.append(self.Line(pt1,pt2,i))
        first_pt,last_pt = contour[0,:],contour[-1,:]
        lines.append(self.Line(first_pt,last_pt,i+1))
        return lines

    def _order_points(self,pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")

    def _get_min_outer_rect(self,contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = self._order_points(box)
        return box

    def _get_max_inner_rect(self, cont, cX, cY, x_min, x_max, y_min, y_max, img=None):
            """中心延展法获取轮廓最大内接正矩形
            cont: 轮廓
            cX,cY: 中心点
            x_min, x_max, y_min, y_max: 轮廓的最大外接矩形的四个顶点
            """
            # img 对应的是原图, 四个极值坐标对应的是最大外接矩形的四个顶点
            c = cont  # 单个轮廓
            # print(c)
            range_x, range_y = x_max - x_min, y_max - y_min   # 轮廓的X，Y的范围
            x1, x2, y1, y2 = cX, cX, cY, cY     # 中心扩散矩形的四个顶点x,y
            cnt_range, radio = 0, 0
            shape_flag = 1                      # 1：轮廓X轴方向比Y长；0：轮廓Y轴方向比X长
            if range_x > range_y:                     # 判断轮廓 X方向更长
                radio, shape_flag = int(range_x / range_y), 1
                range_x_left = cX - x_min
                range_x_right = x_max - cX
                if range_x_left >= range_x_right:   # 取轴更长范围作for循环
                    cnt_range = int(range_x_left)
                if range_x_left < range_x_right:
                    cnt_range = int(range_x_right)
            else:                                   # 判断轮廓 Y方向更长
                radio, shape_flag = int(range_y / range_x), 0
                range_y_top = cY - y_min
                range_y_bottom = y_max - cY
                if range_y_top >= range_y_bottom:   # 取轴更长范围作for循环
                    cnt_range = int(range_y_top)
                if range_y_top < range_y_bottom:
                    cnt_range = int(range_y_bottom)
            print("X radio Y: %d " % radio)
            print("---------new drawing range: %d-------------------------------------" % cnt_range)
            flag_x1, flag_x2, flag_y1, flag_y2 = False, False, False, False
            radio = 5       # 暂时设5，统一比例X:Y=5:1 因为发现某些会出现X:Y=4:1, 某些会出现X:Y=5:1
            if shape_flag == 1:
                radio_x = radio - 1
                radio_y = 1
            else:
                radio_x = 1
                radio_y = radio - 1
            for ix in range(1, cnt_range, 1):      # X方向延展，假设X:Y=3:1，那延展步进值X:Y=3:1
                # 第二象限延展
                if flag_y1 == False:
                    y1 -= 1 * radio_y       # 假设X:Y=1:1，轮廓XY方向长度接近，可理解为延展步进X:Y=1:1
                    p_x1y1 = cv2.pointPolygonTest(c, (x1, y1), False)
                    p_x2y1 = cv2.pointPolygonTest(c, (x2, y1), False)
                    if p_x1y1 <= 0 or y1 <= y_min or p_x2y1 <= 0:  # 在轮廓外，只进行y运算，说明y超出范围
                        for count in range(0, radio_y - 1, 1):    # 最长返回步进延展
                            y1 += 1     # y超出, 步进返回
                            p_x1y1 = cv2.pointPolygonTest(c, (x1, y1), False)
                            if p_x1y1 <= 0 or y1 <= y_min or p_x2y1 <= 0:
                                pass
                            else:
                                break
                        # print("y1 = %d, P=%d" % (y1, p_x1y1))
                        flag_y1 = True

                if flag_x1 == False:
                    x1 -= 1 * radio_x
                    p_x1y1 = cv2.pointPolygonTest(c, (x1, y1), False)    # 满足第二象限的要求，像素都在轮廓内
                    p_x1y2 = cv2.pointPolygonTest(c, (x1, y2), False)    # 满足第三象限的要求，像素都在轮廓内
                    if p_x1y1 <= 0 or x1 <= x_min or p_x1y2 <= 0:       # 若X超出轮廓范围
                        # x1 += 1  # x超出, 返回原点
                        for count in range(0, radio_x-1, 1):       #
                            x1 += 1         # x超出, 步进返回
                            p_x1y1 = cv2.pointPolygonTest(c, (x1, y1), False)  # 满足第二象限的要求，像素都在轮廓内
                            p_x1y2 = cv2.pointPolygonTest(c, (x1, y2), False)  # 满足第三象限的要求，像素都在轮廓内
                            if p_x1y1 <= 0 or x1 <= x_min or p_x1y2 <= 0:
                                pass
                            else:
                                break
                        # print("x1 = %d, P=%d" % (x1, p_x1y1))
                        flag_x1 = True              # X轴像左延展达到轮廓边界，标志=True
                # 第三象限延展
                if flag_y2 == False:
                    y2 += 1 * radio_y
                    p_x1y2 = cv2.pointPolygonTest(c, (x1, y2), False)
                    p_x2y2 = cv2.pointPolygonTest(c, (x2, y2), False)
                    if p_x1y2 <= 0 or y2 >= y_max or p_x2y2 <= 0:  # 在轮廓外，只进行y运算，说明y超出范围
                        for count in range(0, radio_y - 1, 1):  # 最长返回步进延展
                            y2 -= 1     # y超出, 返回原点
                            p_x1y2 = cv2.pointPolygonTest(c, (x1, y2), False)
                            if p_x1y2 <= 0 or y2 >= y_max or p_x2y2 <= 0:  # 在轮廓外，只进行y运算，说明y超出范围
                                pass
                            else:
                                break
                        # print("y2 = %d, P=%d" % (y2, p_x1y2))
                        flag_y2 = True              # Y轴像左延展达到轮廓边界，标志=True
                # 第一象限延展
                if flag_x2 == False:
                    x2 += 1 * radio_x
                    p_x2y1 = cv2.pointPolygonTest(c, (x2, y1), False)    # 满足第一象限的要求，像素都在轮廓内
                    p_x2y2 = cv2.pointPolygonTest(c, (x2, y2), False)    # 满足第四象限的要求，像素都在轮廓内
                    if p_x2y1 <= 0 or x2 >= x_max or p_x2y2 <= 0:
                        for count in range(0, radio_x - 1, 1):  # 最长返回步进延展
                            x2 -= 1     # x超出, 返回原点
                            p_x2y1 = cv2.pointPolygonTest(c, (x2, y1), False)  # 满足第一象限的要求，像素都在轮廓内
                            p_x2y2 = cv2.pointPolygonTest(c, (x2, y2), False)  # 满足第四象限的要求，像素都在轮廓内
                            if p_x2y1 <= 0 or x2 >= x_max or p_x2y2 <= 0:
                                pass
                            elif p_x2y2 > 0:
                                break
                        # print("x2 = %d, P=%d" % (x2, p_x2y1))
                        flag_x2 = True
                if flag_y1 and flag_x1 and flag_y2 and flag_x2:
                    print("(x1,y1)=(%d,%d)" % (x1, y1))
                    print("(x2,y2)=(%d,%d)" % (x2, y2))
                    break
            x1, x2, y1, y2 = int(x1),int(x2),int(y1),int(y2)
            # cv.line(img, (x1,y1), (x2,y1), (255, 0, 0))
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, 8)

            return x1, x2, y1, y2


    def shrink_contour(self,contour,img_shape=None,scale=0.99,pixel=None):
        '''缩小轮廓区域
            Arguments:
                contour: 扶梯梯路区域轮廓
                img_shape: 轮廓对应的图片的形状[H,W]
                scale: 缩小比例, list|tuple|int, e.g. [0.85,0.8], 对于x方向和y方向
                pixel: 缩小的像素, 同scale, 区别是单位为像素级别, 如[10,10]表示x,y都缩小10像素
            Return:
                approx: 缩小后的轮廓
        '''
        assert 0.5<scale[0]<1.2 and 0.5<scale[1]<1.2 if isinstance(scale,(tuple,list)) else \
            0.5<scale<1.2, f'Scale range should be [0.7,1.2]'
        if img_shape is None:
            img_shape = self.img_shape
        mask = np.zeros(img_shape).astype(np.uint8)
        mask_fill = cv2.fillConvexPoly(mask.copy(),contour,color=255)
        # m = cv2.moments(contour)
        # cX = int(m["m10"] / m["m00"])
        # cY = int(m["m01"] / m["m00"])
        x,y,w,h = cv2.boundingRect(contour)
        cX = int(x + w/2)
        cY = int(y + h/2)
        area = cv2.contourArea(contour)
        max_area = 1.5**2 * area
        min_area = 0.5**2 * area

        if pixel is not None:
            if isinstance(pixel,(tuple,list)):
                scalex = 1-pixel[0]/w
                scaley = 1-pixel[0]/h
            else:
                scalex = scaley= 1-pixel/h
        else:
            if isinstance(scale,(tuple,list)):
                scalex,scaley = scale
            else:
                scalex = scaley = scale

        crop = mask_fill[y:y+h,x:x+w]
        crop = cv2.resize(crop,dsize=(0,0),fx=scalex,fy=scaley)
        nh,nw = crop.shape[:2]
        y1 = int(cY-nh/2)
        y2 = int(cY+nh/2)
        x1 = int(cX-nw/2)
        x2 = int(cX+nw/2)
        mask[y1:y2,x1:x2] = crop
        mask = mask.astype(np.uint8)
        # cv2.imshow('test',mask)
        # cv2.waitKey(0)
        # exit()
        new_contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if new_contours:
            areas = [cv2.contourArea(cnt) for cnt in new_contours]
            areas_ids = np.array([(j,area) for j,area in enumerate(areas) if min_area<area<max_area]) # filter
            areas = areas_ids[:,1]
            indexes = areas_ids[:,0]
            idx = int(indexes[np.argmax(areas)]) 
            new_contour = new_contours[idx] # select contour with max area 
            epsilon = 0.005 * cv2.arcLength(new_contour, True)
            approx = cv2.approxPolyDP(new_contour,epsilon,True) # smoothing
            # approx = cv2.convexHull(contour)
            approx = approx.reshape(-1,2)
        else:
            raise AssertionError("No contour has been found")
        return approx

    def part_contour(self,contour,rate=0.5,rect=False):
        '''按比例获取轮廓的下半部分\n
            Arguments:
                contour: 扶梯梯路区域轮廓
                rate: 轮廓截取比例, 0~1, 1表示保留全部轮廓, 从下往上截取
                rect: 是否返回矩形轮廓, 该矩形轮廓为截取后的轮廓的最大内接矩形
            Return:
                new_contour: 截取后的轮廓
        '''
        box = self._get_min_outer_rect(contour)
        tl, tr, br, bl = box
        height = bl[1]-tl[1]
        y_clip = bl[1]-int(height*rate)
        lines = [line for line in self.lines if line.is_inrange(y=y_clip)]
        contour_list = contour.tolist()
        insert_ids = []
        for line in lines:
            x = int(line(y=y_clip))
            if x is not None:
                extra = len(np.array(insert_ids)>=line.id+1)
                contour_list.insert(line.id+1+extra,[x,y_clip])
                insert_ids.append(line.id+1)

        new_contour = []
        for pt in contour_list:
            if pt[1]>=y_clip:
                new_contour.append(pt)
        new_contour = np.array(new_contour)

        if rect:
            m = cv2.moments(new_contour)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            tl, tr, br, bl = self._get_min_outer_rect(new_contour)
            x_min,x_max,y_min,y_max = tl[0],br[0],tl[1],br[1]
            x1, x2, y1, y2 = self._get_max_inner_rect(new_contour,cX,cY,x_min,x_max,y_min,y_max)
            new_contour = np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

        return new_contour

    def part_contour_simple(self,contour,rate=0.5,rect=False):
        '''按比例获取轮廓的下半部分\n
            Arguments:
                contour: 扶梯梯路区域轮廓
                rate: 轮廓截取比例, 0~1, 1表示保留全部轮廓, 从下往上截取
                rect: 是否返回矩形轮廓, 该矩形轮廓为截取后的轮廓的最大内接矩形
            Return:
                new_contour: 截取后的轮廓
        '''
        area = cv2.contourArea(contour)
        max_area = (rate+0.2) * area
        min_area = (rate-0.2) * area

        box = self._get_min_outer_rect(contour)
        tl, tr, br, bl = box
        height = br[1]-tl[1]
        y_clip = br[1]-int(height*rate)

        mask_zero = np.zeros(self.img_shape).astype(np.uint8)
        mask_fill = cv2.fillConvexPoly(mask_zero.copy(),contour,255)
        nx = np.arange(self.img_shape[1])
        ny = np.arange(self.img_shape[0])
        grid_x,grid_y = np.meshgrid(nx,ny)
        mask = np.where(grid_y>=y_clip,mask_fill,mask_zero)

        # vis_img = mask_zero+200
        # self.draw_point(vis_img,box,vis_ptid=True)
        # cv2.imshow('test',vis_img)
        # cv2.waitKey(0)

        new_contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if new_contours:
            areas = [cv2.contourArea(cnt) for cnt in new_contours]
            areas_ids = np.array([(j,area) for j,area in enumerate(areas) if min_area<area<max_area]) # filter
            areas = areas_ids[:,1]
            indexes = areas_ids[:,0]
            idx = int(indexes[np.argmax(areas)]) 
            new_contour = new_contours[idx] # select contour with max area 
            epsilon = 0.005 * cv2.arcLength(new_contour, True)
            new_contour = cv2.approxPolyDP(new_contour,epsilon,True) # smoothing
            # approx = cv2.convexHull(contour)
            new_contour = new_contour.reshape(-1,2)
        else:
            raise AssertionError("No contour has been found")

        if rect:
            m = cv2.moments(new_contour)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            tl, tr, br, bl = self._get_min_outer_rect(new_contour)
            x_min,x_max,y_min,y_max = tl[0],br[0],tl[1],br[1]
            x1, x2, y1, y2 = self._get_max_inner_rect(new_contour,cX,cY,x_min,x_max,y_min,y_max)
            new_contour = np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
        
        # cv2.imshow('test',mask_fill)
        # cv2.waitKey(0)
        # cv2.imshow('test',mask)
        # cv2.waitKey(0)

        return new_contour

    def draw_point(self,img,points,vis_ptid=True):
        for i,pt in enumerate(points):
            cv2.circle(img,(int(pt[0]),int(pt[1])),5,(0,255,255),-1)
            cv2.putText(img,str(i),(int(pt[0])+10,int(pt[1])+10),cv2.LINE_AA,0.75,(55,55,55))
        

def roi_from_json(file_path, part_rate=0.7, reduce_scale=[0.85,0.8], rect=False):
    '''从分割算法保存的json文件中获取光流所需ROI区域
        Arguments:
            file_path: json 文件路径
            part_rate: 轮廓截取比例
            reduce_scale: 截取后的轮廓内缩比例(不做轮廓缩小可能会影响光流特征点的获取)
            rect: 是否返回矩形的ROI区域, default: False
        Return:
            part_contour: 光流法所需取的ROI区域
    '''
    assert os.path.exists(file_path),f'file not found: `{file_path}`'
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    assert 'step' in data, 'step contours not found'
    assert 'imgHeight' in data and 'imgWidth' in data, 'img shape info not found'
    step_contour = np.array(data['step'])
    height = data['imgHeight']
    width = data['imgWidth']

    roi_get = ROI_GETER(step_contour,(height,width))
    part_contour = roi_get.part_contour_simple(step_contour,rate=part_rate,rect=rect)
    if reduce_scale is not None:
        part_contour = roi_get.shrink_contour(part_contour,(height,width),scale=reduce_scale)
    
    return part_contour
        
if __name__ == "__main__":
    track = OpticalTrack()
    camera_id = "data/test.mp4"
    # roi = np.array([[665,447],[845,447],[845,510],[665,510]])
    roi = roi_from_json('config/seg_result.json',part_rate=0.5,rect=False)
    print(roi)
    track.opticalTrack(camera_id,roi)

    # track = OpticalTrack()
    # camera_id = r"../../7月22阶梯型扶梯数据/4mm_202207221514_下行.mp4"
    # roi = roi_from_json(r'../segmentation/output/seg_result.json',part_rate=0.5,rect=False)
    # print(roi)
    # track.opticalTrack(camera_id,roi)

