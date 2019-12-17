from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import csv
#from skimage.morphology import reconstruction

from utils import utils, helpers
from builders import model_builder
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default="checkpoint_", required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=1440, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=1920, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="MobileUNet-Skip", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--checkpoint_foldername', type=str, default="0", help='create folder checkpoint_somenumber')
parser.add_argument('--output_foldername', type=str, default="00", help='create folder output folder')
parser.add_argument('--epoch_number',type=str,default="0001")

args = parser.parse_args()
#model_checkpoint_name = "checkpoint_" + args.checkpoint_foldername + "/latest_model_" + "MobileUNet" + "_" + args.dataset + ".ckpt"
model_checkpoint_name =  "checkpoint_" + args.checkpoint_foldername + "/" + args.epoch_number + "/model.ckpt"
print(model_checkpoint_name)
# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

def postproc(a):
    Kernel = np.ones((7,7),np.uint8)
    a=np.uint8(a)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    Kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        
    for i in range(5):
        a1=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,Kernel2)
        a1=cv2.morphologyEx(a1,cv2.MORPH_OPEN,Kernel2)
    
    #im_floodfill = a1.copy()
    #im_floodfill[a1 == 255] = 0
    #h, w = im_floodfill.shape[:2]
    #mask = np.zeros((h + 2, w + 2), np.uint8)
    #cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    #im_2 = cv2.bitwise_not(im_floodfill)
    #im_2 = cv2.bitwise_not(im_2)

    
    label_values1 = [0,117,29,76,255]
    a1_hot = helpers.reverse_one_hot(helpers.one_hot_it1(a1, label_values1))
    #im2_hot = helpers.reverse_one_hot(helpers.one_hot_it1(im_2, label_values1))
    #dilated1 = reconstruction(im2_hot, a1_hot, method='dilation')

    d = a.copy()
    d[a1_hot==3]=[0,0,255]
    #d[dilated1==4]=[255,0,255]
    d[a1_hot==1]=[255,150,0]
    d[a1_hot==4]=[255,255,255]
    d[a1_hot==2]=[255,0,0]
    return d

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s_%s"%("Test",args.output_foldername)):
        os.makedirs("%s_%s"%("Test",args.output_foldername))

target=open("%s_%s/test_scores.csv"%("Test",args.output_foldername),'w')
target.write("image_name, accuracy, precision, recall, f1 score, f1_wob, Ben_prec, Mal_prec, Iso_prec, Backg_prec, Ben_rec, Mal_rec, Iso_rec,  Backg_rec, Ben_f1, Mal_f1, Iso_f1, Backg_f1 \n")

scores_list = []
precision_list = []
recall_list = []
f1_list = []
f1wob_list = []
run_times_list = []
class_prec_list = []
class_rec_list = []
class_f1_list = []
Sum=0
# Run testing on ALL test images
st1 = time.time()
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1,len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
    #print('--------------gt.shape= ',gt.shape)
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_times_list.append(time.time()-st)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    
    
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    #code for post processing
    #out_vis_image = postproc(out_vis_image)
    #output_image=helpers.reverse_one_hot(helpers.one_hot_it(out_vis_image, label_values))
    
    y_test=gt.reshape(1,gt.shape[0]*gt.shape[1])
    y_test=np.squeeze(y_test)
            
               
    y_pred=output_image.reshape(1,gt.shape[0]*gt.shape[1])
    y_pred= np.squeeze(y_pred)
           
        
    accuracy, prec, rec, f1, f1_wob, class_prec, class_rec, class_f1, Conf_Mat = utils.evaluate_segmentation3(pred=output_image, label=gt, num_classes=num_classes)
    Sum=Sum+Conf_Mat
    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f" % (file_name, accuracy, prec, rec, f1, f1_wob))
    for item in class_prec:
        target.write(", %0.3f" % (item))
    for item in class_rec:
        target.write(", %0.3f" % (item))
    for item in class_f1:
        target.write(", %0.3f" % (item))
    target.write("\n")

    scores_list.append(accuracy)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    f1wob_list.append(f1_wob)
    
    
    class_prec_list.append(class_prec)
    class_rec_list.append(class_rec)
    class_f1_list.append(class_f1)
    gt = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s_%s/%s_pred.png"%("Test",args.output_foldername, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s_%s/%s_gt.png"%("Test",args.output_foldername, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


target.close()
#np.nanmean(np.where(matrix!=0,matrix,np.nan),1)
avg_scores = np.mean(scores_list)
class_avg_scores = np.mean(class_prec_list, axis=0)
class_avg_rec = np.mean(class_rec_list, axis=0)
class_avg_f1 = np.mean(class_f1_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_f1wob = np.mean(f1wob_list)

avg_time = np.mean(run_times_list)
print("Average test accuracy = ", avg_scores)
print("Average per class test accuracies = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Average precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average run time = ", avg_time)


csvData = ['True, predicted-->.', 'benign', 'malignant', 'cytoplasm', 'inflamatory', 'white','recall']
#recall_benign=class_avg_rec[0]
#prec_benign=class_avg_scores[0]
#recall_malignant=class_avg_rec[1]
#prec_malignant=class_avg_scores[1]
#recall_cytoplasm=class_avg_rec[2]
#prec_cytoplasm=class_avg_scores[2]
#recall_inflamatory=class_avg_rec[3]
#prec_inflamatory=class_avg_scores[3]
#recall_white=class_avg_rec[4]
#prec_white=class_avg_scores[4]

recall_benign=Sum[0][0]/(sum(Sum[0,:]))
prec_benign=Sum[0][0]/(sum(Sum[:,0]))
recall_malignant=Sum[1][1]/(sum(Sum[1,:]))
prec_malignant=Sum[1][1]/(sum(Sum[:,1]))
recall_isolated=Sum[2][2]/(sum(Sum[2,:]))
prec_isolated=Sum[2][2]/(sum(Sum[:,2]))
recall_white=Sum[3][3]/(sum(Sum[3,:]))
prec_white=Sum[3][3]/(sum(Sum[:,3]))


recall=[recall_benign, recall_malignant, recall_isolated,  recall_white]
prec=['precision', prec_benign, prec_malignant, prec_isolated, prec_white]
confusion_mat_file="%s_%s/%s.csv"%("Test",args.output_foldername,'confusion_matrix')
with open(confusion_mat_file,'w') as csvFile:
     writer=csv.writer(csvFile)
     writer.writerow(csvData)
     j=1
     for row in Sum:
         cs=[]
         cs.append(csvData[int(j)])
         for i in row:
             cs.append(i)
         cs.append(recall[j-1])
         writer.writerow(cs)
         j=1+j
     writer.writerow(prec)

print('\n--------confusion matrix is \n {}'.format(Sum))
print('\n-------sumof elements is ',np.sum(Sum))


F_score_mal = 2*prec_malignant*recall_malignant/(prec_malignant+recall_malignant)
F_score_ben = 2*prec_benign*recall_benign/(prec_benign+recall_benign)
Dice_score= 2*(Sum[0][0]+Sum[1][1]+Sum[2][2])/((sum(Sum[0,:])-Sum[0,3])+(sum(Sum[:,0])-Sum[3,0])+(sum(Sum[1,:])-Sum[1,3])+(sum(Sum[:,1])-Sum[3,1])+(sum(Sum[2,:])-Sum[2,3])+(sum(Sum[:,2])-Sum[3,2]))



target1=open("%s_%s.csv"%("Test",args.output_foldername),'w')
target1.write("avg scores, recall benign, precision benign, recall malignant, precision malignant, F score_Malignant, F score_Benign, F_Score_wob \n")
target1.write("%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f"%(avg_scores, recall_benign, prec_benign, recall_malignant, prec_malignant, F_score_mal,F_score_ben, Dice_score ))
target1.close()
print("Total time taken is",time.time() -st1)

