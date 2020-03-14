import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
	# Determine the (x, y)-coordinates of the intersection
    x_A = max(prediction_box[0], gt_box[0])
    y_A = max(prediction_box[1], gt_box[1])
    x_B = min(prediction_box[2], gt_box[2])
    y_B = min(prediction_box[3], gt_box[3])

    # Compute intersection
    inter_area = max(0, x_B - x_A) * max(0, y_B - y_A)
    
    # Compute the area of both the prediction and ground-truth rectangles
    prediction_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
	
    # Compute union
    union_area = prediction_area + gt_area - inter_area

    # Compute the intersection over union by taking the intersection
    iou = inter_area / float(union_area)

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1.0
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0.0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # NOTE: NB! Two different ground truths can be matched with the same prediction with this implementation

    # Find all possible matches with a IoU >= iou threshold
    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold

    matched_pred_boxes = []
    matched_gt_boxes = []
    
    for gt_box in gt_boxes:
        best_iou = 0
        best_pred_box = None
        
        # Find valid and best matching
        for prediction_box in prediction_boxes:
            iou = calculate_iou(prediction_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_pred_box = prediction_box
                best_iou = iou
    
        if best_pred_box is not None:
            matched_pred_boxes.append(best_pred_box)
            matched_gt_boxes.append(gt_box)

    return np.array(matched_pred_boxes), np.array(matched_gt_boxes)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # Finds all possible matches for the predicted boxes to the ground truth boxes
    matched_pred, matched_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    # Compute TP, FP and FN
    num_TP = matched_pred.shape[0]
    num_FP = prediction_boxes.shape[0] - num_TP
    num_FN = gt_boxes.shape[0] - num_TP
    
    return  {"true_pos": num_TP, "false_pos": num_FP, "false_neg": num_FN}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    num_TP = 0
    num_FP = 0
    num_FN = 0

    # Loop through all images and calculate precision and recall 
    for pred_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        individual_results = calculate_individual_image_result(pred_boxes, gt_boxes, iou_threshold) 
        num_TP += individual_results["true_pos"]
        num_FP += individual_results["false_pos"]
        num_FN += individual_results["false_neg"]

    precision = calculate_precision(num_TP, num_FP, num_FN)
    recall = calculate_recall(num_TP, num_FP, num_FN)

    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precision = []
    recall = []   
    for c_t in confidence_thresholds:
        img_pred_array = []
        for img_num, pred_boxes in enumerate(all_prediction_boxes):
            predictions = []
            for box_num, pred_box in enumerate(pred_boxes):

                if confidence_scores[img_num][box_num] >= c_t:
                    predictions.append(pred_box)

            # Convert list to numpy array
            predictions = np.array(predictions)      
            
            img_pred_array.append(predictions)
        
        # Convert list to numpy array
        img_pred_array = np.array(img_pred_array) 

        precision_and_recall = calculate_precision_recall_all_images(img_pred_array, all_gt_boxes, iou_threshold)

        # Add precision and recall        
        precision.append(precision_and_recall[0])
        recall.append(precision_and_recall[1])
      
    return np.array(precision), np.array(recall)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    
    max_precisions = [] 
    for recall_level in recall_levels:
        max_precision = 0
        for precision, recall in zip(precisions, recalls):
            if recall >= recall_level and precision >= max_precision:
                max_precision = precision
        max_precisions.append(max_precision)
    
    mAP = np.average(max_precisions)
    return mAP


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
