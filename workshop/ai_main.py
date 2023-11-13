from workshop import ai_03_face_detection_cnn
from workshop import ai_04_face_recognition_cnn
import ai_04_face_recognition_vgg16_cnn
import face_detection_cnn_utils

import logging,os,argparse,re,json,numpy
import torch

from datetime import datetime


def createModel(modelName):
    modelFactory = [ai_03_face_detection_cnn.createModel,
                    ai_04_face_recognition_cnn.createModel,
                    ai_04_face_recognition_vgg16_cnn.createModel]

    for creator in modelFactory:
        modelAndFuncs = creator(modelName)
        if modelAndFuncs:
            return modelAndFuncs
    return None


def run_Train(modelPath, srcPath, epoch, modelName):
    logging.info('train, srcPath=%s,model=%s,epoch=%d' %
                 (srcPath, modelPath, epoch))

    cnn,funcs = createModel(modelName)
    if os.path.exists(modelPath):
        logging.info('loading pre-trained=%s,' %(modelPath))
        cnn.load_state_dict(torch.load(modelPath))

    funcs.train(cnn, srcPath, epoch)

    modelPath = '%s.pt'%(modelName)

    torch.save(cnn.state_dict(), modelPath)

    logging.info('Best model, %s' % modelPath)

    return cnn,modelPath

def run_test(modelPath,srcPath,modelName):
  
    logging.info('test, srcPath=%s,model=%s' % (srcPath,modelPath))

    cnn,funcs = createModel(modelName)
    cnn.load_state_dict(torch.load(modelPath))

    embeddingPath = modelPath+'.embedding'
    embeddings = load_embedding(embeddingPath)

    results = funcs.test(cnn, srcPath, embeddings)

    success = 0
    for imagePath,name,(predict,similarity) in results:
        success += 1 if name==predict else 0
        if name==predict:
            logging.info('%s,   %s, %s, %0.3f, true <====='%(imagePath,name,predict,similarity))
        else:
            logging.info('%s,   %s, %s, %0.3f, false'%(imagePath,name,predict,similarity))

    logging.info('test result,faceModel=%s, testData=%s,success-rate=%%%0.2f'%(modelPath,srcPath,100*success/len(results)))

def extractModelInfoFromFileName(modelPath):
    x = re.search(r'.*v(\d+)-e=(\d+)-l=(\d+\.\d+)',modelPath)
    if x != None:
        groups = x.groups()
        if len(groups) == 3:
            return int(groups[0]),int(groups[1]),float(groups[2])

    return 0,0,0.

def run_embedding(modelPath,srcPath,modelName):

    logging.info('embedding, srcPath=%s,model=%s' % (srcPath,modelPath))

    cnn,funcs = createModel(modelName)
    cnn.load_state_dict(torch.load(modelPath))
 
    allResults = funcs.embedding(cnn, srcPath)

    def toDict(name,embedding):
        return {"name":name,
                "embedding":embedding.tolist()}

    allResults = [toDict(name,embedding) for name,embedding in allResults]

    embeddingPath = modelPath+'.embedding'
    with open(embeddingPath,'w',encoding='utf-8') as f:
        d = {
            'model': modelPath,
            'embeddings':allResults
        }
        f.write(json.dumps(d,indent=4))
    
    logging.info('embedding, created, %s, %d embeddings' % (embeddingPath,len(allResults)))

    return embeddingPath

def load_embedding(embeddingPath):
    with open(embeddingPath, 'r', encoding='utf-8') as f:
        allResults = json.load(f)

    allResults = [(kv['name'], kv['embedding'])
                  for kv in allResults['embeddings']]
    logging.info('load_embedding, embeddingFile=%s,%d names' %
                 (embeddingPath, len(allResults)))
    
    return allResults

def run_similar(modelPath, sourceObject,modelName):

    sourceName = sourceObject if type(sourceObject) is str else 'ImageData'

    logging.info('run_similar, srcPath=%s,model=%s' %
                 (sourceName, modelPath))

    cnn,funcs = createModel(modelName)
    cnn.load_state_dict(torch.load(modelPath))

    embeddingPath = modelPath+'.embedding'
    embeddings = load_embedding(embeddingPath)

    result = funcs.similarity(
        cnn, embeddings, sourceObject, top=3)

    logging.info('==> %s :' % (sourceName))
    for name, similarity in result:
        logging.info('Similarity[%s], similarity=%0.3f' % (name, similarity))

    return result

def run_segment(modelPath, srcImagePath,modelName):

    cnn,funcs = createModel(modelName)
    cnn.load_state_dict(torch.load(modelPath))
    img = funcs.segment(cnn, srcImagePath)
    img.show()

#
bestModel = None
bestModelLoss = 9999999.0
bestModelEpoch = 0
lastAccuracy = 0


def initLogger(logPath = 'log'):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)  # 可以不设，默认是WARNING级别

    formatter = logging.Formatter(
        '%(asctime)s %(filename)s:%(funcName)s:%(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not os.path.exists(logPath):
        os.makedirs(logPath)

    fileHandler = logging.FileHandler(
        "%s/%s.txt" % (logPath,datetime.now().strftime('%Y-%m-%d %H-%M-%S')))
    fileHandler.setLevel(logging.DEBUG)  # 可以不设，默认是WARNING级别
    fileHandler.setFormatter(formatter)  # 设置文件的log格式

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.INFO)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    logging.getLogger('PIL').setLevel(logging.WARNING)


if __name__ == '__main__':

    # Read in build options from the command line
    parser = argparse.ArgumentParser(
        description='Run Webex Feature Toggle Retirement Auto Classification')

    parser.add_argument('--train', action='store_true',
                        help='Run in training mode')

    # parser.add_argument('--retrain', action='store_true',
    #                     help='Run in training mode')
    
    # parser.add_argument('--updateTest', action='store_true',
    #                     help='Run in training mode')

    parser.add_argument('--test', action='store_true',
                        help='Run test')

    parser.add_argument('--embedding', action='store_true',
                        help='Run test')
    
    parser.add_argument('--embeddingPath',
                        help='Run test')    

    parser.add_argument('--similar',action='store_true',
                    help='Run test')   

    parser.add_argument('--segment',action='store_true',
                    help='Show segment')  
    
    parser.add_argument('--srcPath',default='data/train',
                        help='Souce Path of input')

    parser.add_argument('--destPath',default='',
                        help='Evaluation result folder')

    parser.add_argument('--epoch', default=100, type=int,
                        help='Show report')

    parser.add_argument('--modelPath', default='',
                        help='model file path')
    
    parser.add_argument('--segModel', default='',
                        help='Segment model file path')

    parser.add_argument('--modelName',
                    help='Model name')
    
    args = parser.parse_args()

    logFolder = 'log'
    if args.train:
        logFolder = os.path.join(logFolder,'train-%s'%(args.modelName))
    elif args.test:
        logFolder = os.path.join(logFolder,'test-%s'%(args.modelName))
    elif args.embedding:
        logFolder = os.path.join(logFolder,'embedding-%s'%(args.modelName))
    elif args.similar:
        logFolder = os.path.join(logFolder,'similar-%s'%(args.modelName))
    elif args.segment:
        logFolder = os.path.join(logFolder,'segment-%s'%(args.modelName))

    initLogger(logFolder)

    cuda = torch.cuda.is_available()
    cudaDeviceCount = torch.cuda.device_count()
    logging.info("cuda available=%d,device-count=%d"%(cuda,cudaDeviceCount))

    if args.train:
        run_Train(args.modelPath,args.srcPath,args.epoch,args.modelName)
    elif args.test:
        run_test(args.modelPath,args.srcPath,args.modelName)       
    elif args.embedding:
        run_embedding(args.modelPath,args.srcPath,args.modelName)
    elif args.segment:
        run_segment(args.modelPath,args.srcPath,args.modelName)
        
    logging.info('Done')
