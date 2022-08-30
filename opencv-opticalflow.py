import cv2 
import time   
import numpy as np
from tqdm import tqdm
from collections import Counter


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
            

        
if __name__ == "__main__":
    track = OpticalTrack()
    # camera_id = "test.mp4"
    # roi = np.array([[550,328],[670,328],[670,488],[550,488]])
    camera_id = "../4mm_up.mp4"
    # camera_id= "../stop.mp4"
    roi = np.array([[682,385],[790,385],[790,515],[681,514]])

    # camera_id = "../4mm_202207221514_下行.mp4"
    # roi = np.array([[582,420],[710,420],[710,530],[581,529]])
    track.opticalTrack(camera_id,roi)