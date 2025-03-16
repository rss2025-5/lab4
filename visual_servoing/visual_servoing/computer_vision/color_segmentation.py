import cv2
import numpy as np
import csv
import ast

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def iou_score(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Each bbox is ((xmin, ymin), (xmax, ymax)).
    """
    if bbox1[0][0] > bbox1[1][0] or bbox1[0][1] > bbox1[1][1]:
        print ("Check that you are returning bboxes as ((xmin, ymin),(xmax,ymax))")
    # Determine intersection rectangle
    x_int_1 = max(bbox1[0][0], bbox2[0][0])
    y_int_1 = max(bbox1[0][1], bbox2[0][1])
    x_int_2 = min(bbox1[1][0], bbox2[1][0])
    y_int_2 = min(bbox1[1][1], bbox2[1][1])

    # Compute area of intersection
    
    # Check if the bounding boxes are disjoint (no intersection)
    if x_int_2 - x_int_1 < 0 or y_int_2 - y_int_1 < 0:
        area_int = 0
    else:
        area_int = (x_int_2 - x_int_1 + 1) * (y_int_2 - y_int_1 + 1)
    
    # Compute area of both bounding boxes
    area_bbox1 = (bbox1[1][0] - bbox1[0][0] + 1) * (bbox1[1][1] - bbox1[0][1] + 1)
    area_bbox2 = (bbox2[1][0] - bbox2[0][0] + 1) * (bbox2[1][1] - bbox2[0][1] + 1)

    # Compute area of union
    area_union = float(area_bbox1 + area_bbox2 - area_int)

    # Compute and return IoU score
    score = area_int / area_union

    # Reject negative scores
    if score < 0:
        score = 0

    return score


def cd_color_segmentation(img, lower_orange = None, upper_orange = None,  template = None):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_orange = np.array([7, 177, 147]) # TUNE: grid search results: 7, 177, 147
	upper_orange = np.array([30, 255, 255]) # TUNE grid search results: 30, 255, 255
	# mask = cv2.inRange(hsv, lower_orange, upper_orange)
	mask = cv2.inRange(hsv, lower_orange, upper_orange)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=2)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		if cv2.contourArea(largest_contour) > 100:
			x, y, w, h = cv2.boundingRect(largest_contour)
			bounding_box = ((x, y), (x + w, y + h))
		else:
			bounding_box = ((0, 0), (0, 0))
	else:
		bounding_box = ((0, 0), (0, 0))

	return bounding_box

def evaluate_on_dataset(csv_file, lower_orange, upper_orange):
    """
    Evaluate the average IoU for the given HSV thresholds on a dataset."
    """
    total_iou = 0
    count = 0
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = row[0]
            bbox_true = ast.literal_eval(row[1])
            img = cv2.imread(img_path)
            if img is None:
                print("Could not load image:", img_path)
                continue
            bbox_est = cd_color_segmentation(img, lower_orange, upper_orange)
            total_iou += iou_score(bbox_est, bbox_true)
            count += 1
    return total_iou / count if count > 0 else 0

def grid_search_hsv(csv_file):
    """
    Grid search over ranges of HSV thresholds. Adjust the ranges as needed.
    Returns the best parameter set (lower, upper) and a list of all results.
    """
    lower_h_range = [7, 8, 9, 10, 11, 12, 13]       # e.g., [0, 5, 10, 15, 20]
    lower_s_range = [177, 178, 179, 180, 181, 182, 183]     # e.g., [100, 150, 200]
    lower_v_range = [147, 148, 149, 150, 151, 152]     # e.g., [100, 150, 200]
    upper_h_range = [28, 29, 30, 31, 32]      # e.g., [10, 15, 20, 25, 30, 35, 40]
    upper_s_range = [250, 251, 252, 253, 254, 255, 256]
    upper_v_range = [250, 251, 252, 253, 254, 255, 256]

    best_score = 0
    best_params = None
    results = []
    
    for lh in lower_h_range:
        for ls in lower_s_range:
            for lv in lower_v_range:
                for uh in upper_h_range:
                    for us in upper_s_range:
                        for uv in upper_v_range:
                            lower = np.array([lh, ls, lv])
                            upper = np.array([uh, us, uv])
                            avg_iou = evaluate_on_dataset(csv_file, lower, upper)
                            results.append(((lh, ls, lv, uh, us, uv), avg_iou))
                            print(f"Params: lower={lower}, upper={upper} -> avg_iou = {avg_iou:.3f}")
                            if avg_iou > best_score:
                                best_score = avg_iou
                                best_params = (lower, upper)
    
    print("Best parameters:", best_params, "with avg IoU =", best_score)
    return best_params, results

if __name__ == "__main__":
	# individual image bbox testing

    img = cv2.imread("test_images_cone/test11.jpg") # change to problem files
    if img is None:
        print("Test image not found!")
    else:
        bbox = cd_color_segmentation(img, None)
        print("Detected bounding box:", bbox)
        
        (x1, y1), (x2, y2) = bbox
        if bbox != ((0, 0), (0, 0)):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Detected Bounding Box", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

	# search for optimal hsv threshold testing

	# csv_file = "test_images_cone/test_images_cone.csv"  # Adjust if necessary
	# best_params, all_results = grid_search_hsv(csv_file)
	# output_file = "hsv_grid_search_results.csv"
	# with open(output_file, "w", newline="") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	writer.writerow(["lower_h", "lower_s", "lower_v", "upper_h", "upper_s", "upper_v", "avg_iou"])
	# 	for params, avg_iou in all_results:
	# 		writer.writerow(list(params) + [avg_iou])