#coding:utf8
import time
import warnings

tfmt = '%m%d_%H%M%S'

class Config(object):
    '''
    并不是所有的配置都生效,实际运行中只根据需求获取自己需要的参数
    '''

    loss = 'multilabel_marginloss'
    loss = 'multilabelloss'
    model='MultiCNNTextBNDeep' 
    model='LSTMText' 
    data_dim = 200 #内容的卷积核数
    num_classes = 202 # 类别
    embedding_dim = 256 # embedding大小
    linear_hidden_size = 1000 # 全连接层隐藏元数目
    kmax_pooling = 2# k
    hidden_size = 256 #LSTM hidden size
    num_layers=2 #LSTM layers
    inception_dim = 512 #inception的卷积核数
    logit_threshold = 0.6
    
    # vocab_size = 11973 # num of chars
    vocab_size = 71462# num of words 
    kernel_size = 3 #单尺度卷积核
    kernel_sizes = [2,3,4] #多尺度卷积核
    data_seq_len = 512 #描述长度 word为120 char为250
    all=False # 模型同时训练char和word
    cate_dist_path='cat_dist.new'
    cate_dist_path='cat_dist.lst'

    embedding_path = 'preprocess/word_embedding.npz' # Embedding
    data_src = 'small'
    data_src = 'big'
    data_tail= '.ph'
    data_tail=''
    train_data_path = 'preprocess/' + data_src + '/train.convrted' + data_tail # train
    test_data_path = 'preprocess/' + data_src + '/test.convrted' + data_tail # train
    validation_data_path = 'preprocess/' + data_src + '/validation.convrted' + data_tail # train

    result_path='csv/'+time.strftime(tfmt)+'.csv'
    shuffle = True # 是否需要打乱数据
    num_workers = 4 # 多线程加载所需要的线程数目
    pin_memory =  True # 数据从CPU->pin_memory—>GPU加速
    batch_size = 128
    batch_size = 200

    env = time.strftime(tfmt) # Visdom env
    plot_every = 100 # 每10个batch，更新visdom等

    max_epoch=100
    lr = 5e-3 # 学习率
    lr2 = 0 # embedding层的学习率
    lr2 = 1e-3 # embedding层的学习率
    min_lr = 1e-5 # 当学习率低于这个值，就退出训练
    lr_decay = 0.99 # 当一个epoch的损失开始上升lr = lr*lr_decay 
    weight_decay = 0 #2e-5 # 权重衰减
    weight = 1 # 正负样本的weight
    decay_every = 1000 #每多少个batch 查看一下score,并随之修改学习率

    model_path = None # 如果有 就加载
    optimizer_path='optimizer.pth' # 优化器的保存地址

    debug_file = '/tmp/debug2' #若该文件存在则进入debug模式
    debug=False

    augument=True # 是否进行数据增强
    static=False
    
    ### multimode 用到的
    model_names=['MultiCNNTextBNDeep','CNNText_inception','RCNN','LSTMText','CNNText_inception']
    model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788','checkpoints/CNNText_tmp_0.380390420742','checkpoints/RCNN_word_0.373609030286','checkpoints/LSTMText_word_0.381833388089','checkpoints/CNNText_tmp_0.376364647145']#,'checkpoints/CNNText_tmp_0.402429167301']

    val=False # 跑测试集还是验证集?
    
def parse(self,kwargs,print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 

        if print_:
            print('user config:')
            print('#################################')
            for k in dir(self):
                if not k.startswith('_') and k!='parse' and k!='state_dict':
                    print k,getattr(self,k)
            print('#################################')
        return self

def state_dict(self):
    return  {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }

Config.parse = parse
Config.state_dict = state_dict
opt = Config()
