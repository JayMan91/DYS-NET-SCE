############################################### Lightning Wrappers to interact pyepo models #######################################

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pyepo
from pyepo import EPO
from datetime import datetime
from pyepo.func.utlis import  regret_loss


class PFL(pl.LightningModule):
    def __init__(self, net,  optmodel, method_name='pfl',  lr=1e-2, max_epochs=30,  
                 scheduler=False, seed=42,  **kwd):
        """
        A class to implement Diffeential Blackbox with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net =  net
        self.method_name = method_name
        self.optmodel = optmodel
        self.save_hyperparameters(  'lr','max_epochs',  'scheduler', 'seed')
        self.start_time = datetime.now()

    def forward(self,x):
        return self.net(x) 
    
    def training_step(self, batch, batch_idx):
        self.net.train()
        criterion = nn.MSELoss(reduction='mean')
        
        x, c, w, z = batch
        method_name = self.method_name
        optmodel = self.optmodel
        cp =  self(x)
        training_loss  = criterion(cp, c)

        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss
    def validation_step(self, batch, batch_idx, testing=False ):
        self.net.eval()
        criterion = nn.MSELoss(reduction='mean')
        x, c, w, z = batch
        
        cp =  self(x)
        # print ("SHAPE inside validation")
        # print ( c.shape, cp.shape, w.shape, z.shape)
        # torch.save(c, 'true_cost.pt')
        # torch.save(cp, 'pred_cost.pt')
        # torch.save(z, 'obj.pt')
        mseloss = criterion(cp, c)
        regret = pyepo.metric.regret(self.net, self.optmodel, batch)
        elapsed = (datetime.now() - self.start_time ).total_seconds() 
        meanabsval = torch.abs(cp).mean()


        if not testing:
            self.log("val_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_meanabsvalue", meanabsval, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_regret", regret, prog_bar=True, on_step=False, on_epoch=True)
            self.log("elapsedtime", elapsed, prog_bar=True, on_step=False, on_epoch=True)
            return {'val_mse': mseloss, 'val_regret': regret}
        else:
            self.log("test_mse", mseloss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("test_regret", regret, prog_bar=True, on_step=False, on_epoch=True)
            self.log("elapsedtime", elapsed, prog_bar=True, on_step=False, on_epoch=True)
            return {'test_mse': mseloss, 'test_regret': regret}

    def test_step(self, batch, batch_idx):
        # Reuse validation_step for testing
        return self.validation_step(batch, batch_idx,  testing=True)
    def on_train_epoch_end(self):
        elapsed = (datetime.now() - self.start_time ).total_seconds() 
        self.log("elapsedtime", elapsed, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        self.net.eval()
        x, c, w, z = batch
        return self(x)

    def configure_optimizers(self):
        ############# Adapted from https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html #############
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler:
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.2,
                patience=2,
                min_lr=1e-6),
                        "monitor": "val_regret"
                    },
                }
        return optimizer



class SPO(PFL):
    def __init__(self, net,  optmodel, optmodel_train= None, method_name='spo', normalize=False,
                    lr=1e-2, max_epochs=30,  scheduler=False, seed=42, dataset=None, solve_ratio=1, 
                    processes=1, **kwd):
        """
        A class to implement SPO with test and validation module
        Args:
            net: the underlying predictive model
            
        """
# net,  optmodel, method_name, normalize,  lr, max_epochs,  scheduler, seed, 
#                          processes = processes , **kwd

        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        if optmodel_train is None:
            optmodel_train = optmodel
        self.spoplus = pyepo.func.SPOPlus(optmodel_train, solve_ratio=solve_ratio, dataset=dataset, processes=processes)
        self.save_hyperparameters( 'solve_ratio' ,'lr','max_epochs',  'scheduler', 'seed', 'normalize')
        self.start_time = datetime.now()

        

    def training_step(self, batch, batch_idx):
        self.net.train()
        x, c, w, z = batch
        method_name = self.method_name
        # optmodel = self.optmodel
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        
        training_loss  = self.spoplus(cp, c, w, z)
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss


class DYS_Active(PFL):
    def __init__(self, net,  optmodel, AbCD_provided= None, method_name='dys_active', inv_provided= False, normalize=False, 
                dflloss = 'regret', alpha = 0.05, tau = 1., cost_transform = None, sol_transform = None,
                num_iter = 100, decay_param = 10,  dopresolve=False,  lr=1e-2, max_epochs=30,  
                scheduler=False, seed=42, processes=1, **kwd):
        """
        A class to implement DYS
        Args:
            net: the underlying predictive model
            
        """
        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        # self.dysopt = pyepo.func.DYSOpt(optmodel, AbCD_provided=  AbCD_provided, num_iter = num_iter, alpha = alpha, tau = tau, decay_param = decay_param , dopresolve = dopresolve,
        #                                 cost_transform = cost_transform, sol_transform = sol_transform)
        self.dys_layer = pyepo.func.DYSOpt_OTF(num_iter = num_iter, alpha = alpha, tau = tau,
                         decay_param = decay_param, processes = processes, inv_provided= inv_provided)
        #pyepo.func.DYS_OTF(num_iter = 50, alpha = 0.05, tau = 1., decay_param = 10)
        self.save_hyperparameters('num_iter','alpha' , 'tau','decay_param', 'lr','max_epochs', 'inv_provided' ,
                                  'dflloss' , 'normalize', 'scheduler', 'seed')

    def training_step(self, batch, batch_idx):  

        dflloss = self.hparams.dflloss


        self.net.train()      
        # x, c, w, z, As, Ainvs = batch
        
        x = batch[0]
        c = batch[1]
        w = batch[2]
        Aeqs = batch[4]
        if self.hparams.inv_provided:
            Aeqs_inv = batch[5]

        # print ('Shape of A-eq', Aeqs.shape)

        batch_size, n_cons, _ = Aeqs.size()
        # As = torch.cat ( (Aeqs, torch.eye(n_cons)), dim=1) 

        batch_size, n_var = c.size()
        # batch_size, n_cons, _ = As.size()
        cp =  self(x)
        device = cp.device

        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        # Aeqs = As[:,:,  :n_var]
        bs = torch.einsum("brc,bc->br",Aeqs, w.float())
        
        # wp =  torch.zeros_like(w).float()
        if dflloss=="SPO":
            c_input = (2*cp -c)
        else:
            c_input = cp
        if self.optmodel.modelSense == EPO.MAXIMIZE:
                c_input = -1*c_input 
        
        c_input = torch.cat ([ c_input, torch.zeros(batch_size, n_cons) ], -1)

        if self.hparams.inv_provided:
             wp = self.dys_layer  (c_input, Aeqs,  bs, Aeqs_inv)
        else:
            wp = self.dys_layer  (c_input, Aeqs,  bs)
        wp = wp [:, :n_var]

        # print ("Solution", w[0])
        # print ("Prediocted Solution", wp[0])
        
        if dflloss=="regret":
            training_loss  = torch.einsum("bd,bd->b", c, wp).mean() #C1
        elif dflloss=="SCE":
            training_loss  = torch.einsum("bd,bd->b", cp, w-wp).mean() #C2
        elif dflloss=="SPO":
            training_loss  = torch.einsum("bd,bd->b", 2*cp  - c, w-wp).mean() #C3
        elif dflloss=="NCEMAP":
            training_loss  = torch.einsum("bd,bd->b", cp -c, w-wp).mean() #C2

        elif dflloss=="Squared":
            training_loss  =  ( (w - wp)**2  ).mean() #C2
        else:
            raise NotImplementedError

        if self.optmodel.modelSense == EPO.MAXIMIZE:
            if dflloss!="Squared":
                training_loss  = -1 * training_loss
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss.mean()




# net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed
class DBB(PFL):
    def __init__(self, net,  optmodel, optmodel_train = None, method_name='dbb',normalize=False, lambda_val=1.,  lr=1e-2, 
                 max_epochs=30,  scheduler=False, seed=42 , processes = 1, **kwd):
        """
        A class to implement Diffeential Blackbox with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        if optmodel_train is None:
            optmodel_train = optmodel
        self.dbb = pyepo.func.blackboxOpt(optmodel_train, lambd= lambda_val, processes= processes)
        self.save_hyperparameters('lambda_val',  'lr','max_epochs',  'scheduler', 'normalize', 'seed')

    def training_step(self, batch, batch_idx):
        self.net.train()
        lambda_val = self.hparams.lambda_val
        
        x, c, w, z = batch
        method_name = self.method_name
        optmodel = self.optmodel
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        
        wp = self.dbb(cp)
        training_loss  = regret_loss(wp,c, self.optmodel.modelSense)

        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss.mean()
 
class CVX(PFL):
    def __init__(self, net,  optmodel, cvxobj = None, build_from_optmodel =True, load_cvxmodel = False, 
                method_name='cvx',normalize=False, dflloss = 'regret', tau = 1.,  
                 regret_withTransformC = False, sol_transform = None,  cost_transform = None,
                 lr=1e-2, max_epochs=30,   scheduler=False, seed=42, processes=1, **kwd):
        """
        A class to implement DYS
        Args:
            net: the underlying predictive model
            
        """
        super().__init__( net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed )

        if load_cvxmodel:
            if cvxobj is None:
                raise RuntimeError("There is no CVX model, cannot load from cvx")

        self.cvxopt = pyepo.func.CVXOpt(build_from_optmodel = build_from_optmodel , load_cvxmodel = load_cvxmodel, 
                    optmodel = optmodel, cvxobj = cvxobj , 
                    regret_withTransformC =regret_withTransformC, cost_transform = cost_transform, sol_transform = sol_transform,
                    tau= tau, processes = processes)
        if regret_withTransformC:
            self.cost_transform = cost_transform

        self.save_hyperparameters('tau', 'regret_withTransformC', 'lr','max_epochs', 'dflloss' , 'normalize', 'scheduler', 'seed')

    def training_step(self, batch, batch_idx): 
        self.net.train()       
        x, c, w, z = batch
        regret_withTransformC = self.hparams.regret_withTransformC
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        if regret_withTransformC:
            transformed_c  =  self.cost_transform.apply (c) 
            transformed_w =  self.cvxopt(c)
            transformed_cp =  self.cost_transform.apply (cp)
        
        dflloss = self.hparams.dflloss

        if not regret_withTransformC:
            ## The usual case
            if dflloss=="regret":
                wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", c, wp).mean() #C1

            elif dflloss=="SPO":
                wp = self.cvxopt(2*cp  - c)
                training_loss  = torch.einsum("bd,bd->b", 2*cp  - c, w-wp).mean() #C3
            elif dflloss=="NCEMAP":
                wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", cp -c, w-wp).mean() #C2

            elif dflloss=="Squared":
                wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", w-wp, w-wp).mean() #C2
            else:
                raise NotImplementedError
            # training_loss  = regret_loss(wp,c, self.optmodel.modelSense)
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                if dflloss!="Squared":
                    training_loss  = -1 * training_loss
        elif regret_withTransformC:
            ### This is for facility location, we should consider the whole cost vector
            if dflloss=="regret":
                transformed_wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_c, transformed_wp).mean() #C1

            elif dflloss=="SPO":
                transformed_wp = self.cvxopt(2*cp  - c)
                training_loss  = torch.einsum("bd,bd->b", 2*transformed_cp  - transformed_c, transformed_w-transformed_wp).mean() #C3
            elif dflloss=="NCEMAP":
                transformed_wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_cp -transformed_c, transformed_w-transformed_wp).mean() #C2
                # print (transformed_c.shape, transformed_w.shape, c.shape)

            elif dflloss=="Squared":
                transformed_wp = self.cvxopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_w-transformed_wp, transformed_w-transformed_wp).mean() #C2
            else:
                raise NotImplementedError
            # training_loss  = regret_loss(wp,c, self.optmodel.modelSense)
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                if dflloss!="Squared":
                    training_loss  = -1 * training_loss    
        
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)

        return training_loss.mean()
 

 
class DYS(PFL):
    def __init__(self, net,  optmodel, AbCd_provided= None, method_name='dys',normalize=False, dflloss = 'regret',
                 regret_withTransformC = False, sol_transform = None,  cost_transform = None,  
                alpha = 0.05, tau = 1., doScale =False, num_iter = 100, decay_param = 10, doQR= False,   dopresolve=False, 
                  lr=1e-2, max_epochs=30,  scheduler=False, seed=42, processes = 1, **kwd):
        """
        A class to implement DYS
        Args:
            net: the underlying predictive model
            
        """
        super().__init__( net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed )
        self.dysopt = pyepo.func.DYSOpt(optmodel, AbCd_provided=  AbCd_provided, num_iter = num_iter, 
                    alpha = alpha, tau = tau, decay_param = decay_param ,
                      dopresolve = dopresolve, doQR = doQR, doScale = doScale,
                      regret_withTransformC =regret_withTransformC, cost_transform = cost_transform, sol_transform = sol_transform,
                     processes = processes)
        if regret_withTransformC:
            self.cost_transform = cost_transform

        self.save_hyperparameters('num_iter','alpha' , 'tau','decay_param',  'regret_withTransformC',
                                  'lr','max_epochs', 'dflloss' , 'normalize', 'scheduler', 'seed')

    def training_step(self, batch, batch_idx):  
        regret_withTransformC = self.hparams.regret_withTransformC
        self.net.train()      
        x, c, w, z = batch
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        # w = F.normalize(w, p=1,dim = 1)
        # wp = self.dysopt(cp)

        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        if regret_withTransformC:
            transformed_c  =  self.cost_transform.apply (c) 
            transformed_w =  self.cvxopt(c)
            transformed_cp =  self.cost_transform.apply (cp)

        dflloss = self.hparams.dflloss
        if not regret_withTransformC:
            if dflloss=="regret":
                wp = self.dysopt(cp)
                training_loss  = torch.einsum("bd,bd->b", c, wp).mean() #C1
            elif dflloss=="SPO":
                wp = self.dysopt(2*cp  - c)
                training_loss  = torch.einsum("bd,bd->b", 2*cp  - c, w-wp).mean() #C3
            elif dflloss=="NCEMAP":
                wp = self.dysopt(cp)
                training_loss  = torch.einsum("bd,bd->b", cp -c, w-wp).mean() #C2
            elif dflloss=="Squared":
                wp = self.dysopt(cp)
                training_loss  =  ( (w - wp)**2  ).mean() #C2
            else:
                raise NotImplementedError
            # training_loss  = regret_loss(wp,c, self.optmodel.modelSense)
        elif regret_withTransformC:
            if dflloss=="regret":
                transformed_wp = self.dysopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_c, transformed_wp).mean() #C1

            elif dflloss=="SPO":
                transformed_wp = self.dysopt(2*cp  - c)
                training_loss  = torch.einsum("bd,bd->b", 2*transformed_cp  - transformed_c, transformed_w-transformed_wp).mean() #C3
            elif dflloss=="NCEMAP":
                transformed_wp = self.dysopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_cp -transformed_c, transformed_w-transformed_wp).mean() #C2

            elif dflloss=="Squared":
                transformed_wp = self.dysopt(cp)
                training_loss  = torch.einsum("bd,bd->b", transformed_w-transformed_wp, transformed_w-transformed_wp).mean() #C2
            else:
                raise NotImplementedError
            # training_loss  = regret_loss(wp,c, self.optmodel.modelSense)
            if self.optmodel.modelSense == EPO.MAXIMIZE:
                if dflloss!="Squared":
                    training_loss  = -1 * training_loss  
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            if dflloss!="Squared":
                training_loss  = -1 * training_loss
        
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)

        return training_loss.mean()




# net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed
class NID(PFL):
    def __init__(self, net,  optmodel, optmodel_train = None, method_name='nid', normalize=False,  lr=1e-2, max_epochs=30,  
                 scheduler=False, seed=42,processes = 1,  **kwd):
        """
        A class to implement Negative Identity Backpropagation with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        if optmodel_train is None:
            optmodel_train = optmodel


        self.nid = pyepo.func.negativeIdentity(optmodel_train, processes = processes)
        self.save_hyperparameters(  'lr','max_epochs', 'normalize',  'scheduler', 'seed')
    
    def training_step(self, batch, batch_idx):
        self.net.train()
        x, c, w, z = batch
        method_name = self.method_name
        optmodel = self.optmodel
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        
        wp = self.nid(cp)
        training_loss  = regret_loss(wp,c, self.optmodel.modelSense)
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss.mean()


class PFY(PFL):
    def __init__(self, net,  optmodel, optmodel_train = None, method_name='pfy',normalize=False,
                   n_samples=3, sigma=1.0,  
                 lr=1e-2, max_epochs=30,  scheduler=False, seed=42, processes = 1, **kwd):
        """
        A class to implement Perturbed Fenchel-Young Loss with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__( net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed )

        if optmodel_train is None:
            optmodel_train = optmodel

        self.pfy = pyepo.func.perturbedFenchelYoung(optmodel_train, n_samples= n_samples, sigma=sigma, processes = processes)
        self.save_hyperparameters( 'n_samples' , 'sigma', 'normalize',  'lr','max_epochs',  'scheduler', 'seed')

    def training_step(self, batch, batch_idx):
        self.net.train()
        n_samples, sigma = self.hparams.n_samples, self.hparams.sigma
  
        x, c, w, z = batch
        method_name = self.method_name
        optmodel = self.optmodel
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        
        training_loss  = self.pfy(cp, w)
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss.mean()


class NCEMAP(PFL):
    ### In pyepo, it's named different

    def __init__(self, net,  optmodel, optmodel_train =None, method_name='cmap', normalize=False,  lr=1e-2, max_epochs=30, 
                  scheduler=False, seed=42, dataset=None, solve_ratio=1., processes = 1, **kwd):
        """
        A class to implement Noise Contrastive Estimation MAP with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        if optmodel_train is None:
            optmodel_train = optmodel
        self.sce_func = pyepo.func.contrastiveMAP(optmodel_train,  solve_ratio=  solve_ratio, dataset=dataset, processes = processes  )
        self.save_hyperparameters( 'solve_ratio' ,   'lr', 'max_epochs', 'normalize',  'scheduler', 'seed')
    

    def training_step(self, batch, batch_idx):
        self.net.train()
   
        x, c, w, z = batch
        method_name = self.method_name
        optmodel = self.optmodel
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        
        training_loss  = self.sce_func (cp, c, w, z)
        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss.mean()

class NCEMAP_Linear(NCEMAP):
    def __init__(self, net,  optmodel, optmodel_train = None, method_name='cmap', normalize=False,   lr=1e-2, max_epochs=30,
                   scheduler=False, seed=42, dataset=None, solve_ratio=1., processes = 1, **kwd):
        """
        A class to implement Noise Contrastive Estimation MAP with test and validation module
        Args:
            net: the underlying predictive model
            
        """
        super().__init__(net,  optmodel, optmodel_train , method_name, normalize,  lr, max_epochs,  scheduler, seed, 
                    dataset , solve_ratio, processes = processes, **kwd)
        self.sce_func = pyepo.func.contrastiveMAP_linear(optmodel_train,  solve_ratio=  solve_ratio,  dataset=dataset, processes = processes)


class CAVE(PFL):
    def __init__(self, net, optmodel, optmodel_train= None,  method_name='cave', solver="clarabel", 
                 normalize=False,  lr=1e-2, max_epochs=30,  scheduler=False, seed=42, dataset=None,
                 cost_transform = None,  sol_transform = None, 
                 max_iter=3, solve_ratio=1,  inner_ratio=0.2, processes=1, **kwd):
        super().__init__(net,  optmodel, method_name,  lr, max_epochs,  scheduler, seed)
        if optmodel_train is None:
            optmodel_train = optmodel
        self.cave  =  pyepo.func.innerConeAlignedCosine(optmodel_train, solver=solver, max_iter=max_iter,
             solve_ratio=solve_ratio, inner_ratio=inner_ratio , processes = processes)
        self.cost_transform = cost_transform
        self.sol_transform = sol_transform
        self.save_hyperparameters(  'lr','max_epochs', 'normalize',  'scheduler', 'seed', 'solver')
    def training_step(self, batch, batch_idx):
        self.net.train()
        x, c, w, z, bctr = batch 
        # print ("bctr shape", bctr.shape, "c shape", c.shape)       
        cp =  self(x)
        if self.hparams.normalize:
            cp = F.normalize(cp, p=1,dim = 1)
            c = F.normalize(c, p=1,dim = 1)
        if self.cost_transform is not None:
            cp  =  self.cost_transform.apply (cp)
        # print ("cps hape", cp.shape)
        training_loss   = self.cave(cp, bctr)

        self.log('train_loss', training_loss, prog_bar=False, on_epoch=True, on_step=False)
        return training_loss