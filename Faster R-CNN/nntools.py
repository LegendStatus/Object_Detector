"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.

Modified by Ziyan Zhu 
"""

import os
import time
import torch
from torch import nn
import torch.utils.data as td

class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update

class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    def __init__(self, net, train_set, val_set, optimizer, stats_manager,
                 output_dir=None, batch_size=1, perform_validation_during_training=False):
        # get device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Define data loaders
        train_loader = td.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = td.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Initialize history
        history = []

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    print(repr(self))
                    print('______________________________________')
                    print(f.read()[:-1])
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Net': self.net,
                'TrainSet': self.train_set,
                'ValSet': self.val_set,
                'Optimizer': self.optimizer,
                #'Scheduler': self.scheduler,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': self.net.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                #'Scheduler': self.scheduler.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        #self.scheduler.load_state_dict(checkpoint['Scheduler'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)

        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()
            
            for images, targets in self.train_loader:
                # move data to device
                images = images.to(self.device)
                targets['boxes'] = targets['boxes'].to(self.device)[0]
                targets['labels'] = targets['labels'].to(self.device)[0]
                # Initialize the gradients to zero
                self.optimizer.zero_grad()
                # Forward propagation and compute loss
                loss_dict = self.net(images,[targets])
                losses = sum(loss for loss in loss_dict.values())
                # Back propagation
                losses.backward()
                # Parameter update
                self.optimizer.step()
                # Compute statistics
                with torch.no_grad():
                    self.stats_manager.accumulate(losses, 0,0,0)
                    
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append((self.stats_manager.summarize(), self.validate()))
            print("Epoch {} (Time: {:.2f}s)".format(
                self.epoch, time.time() - s))
            #self.scheduler.step()
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))
    
    def validate(self):
        self.stats_manager.init()
        self.net.train()
        with torch.no_grad():
            for images, targets in self.val_loader:
                # move data to device
                images = images.to(self.device)
                targets['boxes'] = targets['boxes'].to(self.device)[0]
                targets['labels'] = targets['labels'].to(self.device)[0]

                loss_dict = self.net(images, [targets])
                losses = sum(loss for loss in loss_dict.values())
                self.stats_manager.accumulate(losses,0,0,0)
        return self.stats_manager.summarize()
    
    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        #需要改！！！！！！！！！！！！！！！
        self.stats_manager.init()
        self.net.eval()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # move data to device
                images = images.to(self.device)
                targets['boxes'] = targets['boxes'].to(self.device)[0]
                targets['labels'] = targets['labels'].to(self.device)[0]

                loss_dict = self.net(images, [targets])
                losses = sum(loss for loss in loss_dict.values())
                
                self.stats_manager.accumulate(losses,0,0,0)
        self.net.train()
        return self.stats_manager.summarize()
    
    
    def myfilter(self, bboxes, scores, labels, threshold = 0.7):
        ind = (scores >= threshold).nonzero().squeeze()
        bboxes = bboxes[ind]
        scores = scores[ind]
        labels = labels[ind]
        return bboxes, scores, labels

    def nms(self, bboxes, scores, threshold=0.6):
            _, order = scores.sort(0, descending=True)    # descrasing order

            x1 = bboxes[:,0]
            y1 = bboxes[:,1]
            x2 = bboxes[:,2]
            y2 = bboxes[:,3]
            areas = (x2-x1)*(y2-y1)   # [N,] area of each bbox


            keep = []
            while order.numel() > 0:       # torch.numel() returns the number of elements
                if order.numel() == 1:     # only one left
                    i = order.item()
                    keep.append(i)
                    break
                else:
                    i = order[0].item()    # keep the bbox with largest score
                    keep.append(i)

                # 计算box[i]与其余各框的IOU(思路很好)
                xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
                yy1 = y1[order[1:]].clamp(min=y1[i])
                xx2 = x2[order[1:]].clamp(max=x2[i])
                yy2 = y2[order[1:]].clamp(max=y2[i])
                inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

                iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
                idx = (iou <= threshold).nonzero().squeeze() # 
                if idx.numel() == 0:
                    break
                order = order[idx+1]   
            return torch.LongTensor(keep)   
    
    def AP(self, recall, precision):
        '''Compute average precision for one class'''
        rec = np.concatenate(([0.], recall, [1.]))
        prec = np.concatenate(([0.], precision, [0.]))
        for i in range(prec.size -1, 0, -1):
            prec[i-1] = np.maximum(prec[i-1],prec[i])
        i = np.where(rec[1:] != rec[:-1])[0]
        ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
        return ap
    
    

