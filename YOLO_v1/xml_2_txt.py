import xml.etree.ElementTree as ET
import os
import argparse

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects

if __name__ == "__main__":
    """ Convert xml file to txt with bounding box coordinates and corresponding class label """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Train or Valditation set")
    opt = parser.parse_args()
    mode = opt.mode
    
    txt_file = open('voc2012'+mode+'.txt','w')
    Annotations = './data/Annotations/'
    Main = './data/ImageSets/Main'
    xml_files = os.listdir(Annotations)
    test_file = open(os.path.join(Main, mode+'.txt'),'r') 
    lines = test_file.readlines()
    lines = [x[:-1] for x in lines]
    count = 0
    
    print('Processing')
    for xml_file in xml_files:
        count += 1
        if xml_file.split('.')[0] not in lines:
            continue
        image_path = xml_file.split('.')[0] + '.jpg'
        results = parse_rec(Annotations + xml_file)
        if len(results)==0:
            continue
        txt_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
        txt_file.write('\n')
        
    txt_file.close()
    print('COMPLETE')

