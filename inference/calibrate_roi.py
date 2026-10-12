import argparse
import yaml
import cv2
import numpy as np

pts = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(param['img'], (x, y), 3, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.line(param['img'], pts[-2], pts[-1], (0, 0, 255), 2)
        cv2.imshow('image', param['img'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    
    cap = cv2.VideoCapture(a.source)
    ret, img = cap.read()
    if not ret:
        print("Could not read video")
        return
    cap.release()
    
    img = cv2.resize(img, (1280, 720))
    clone = img.copy()
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, {'img': img})
    
    print("Click points for the hood outline. Press 'q' or 'ENTER' to finish. Press 'c' to clear.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13:
            break
        elif key == ord('c'):
            img = clone.copy()
            cv2.imshow('image', img)
            pts.clear()
    
    cv2.destroyAllWindows()
    
    if len(pts) < 3:
        print("Not enough points to form a polygon. Aborting.")
        return
        
    h, w = img.shape[:2]
    norm_pts = [[round(x / w, 4), round(y / h, 4)] for x, y in pts]
    
    # Write to config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    if 'inference' not in config:
        config['inference'] = {}
    config['inference']['hood_polygon'] = norm_pts
    if 'roi_polygon' in config['inference']:
        del config['inference']['roi_polygon']
        
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False)
        
    print("Saved hood_polygon to config.yaml:", norm_pts)

if __name__ == '__main__':
    main()
