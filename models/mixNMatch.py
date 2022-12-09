import torch
from torch import nn
from models.mixNMatch_utils.train_first_stage import define_optimizers, load_network
from models.mixNMatch_utils.utils import copy_G_params, CrossEntropy, cal_gradient_penalty, child_to_parent
from models.train import LitTrainer

class MixNMatch(LitTrainer):
    def __init__(self, gan_cfg, bg_loss_wt, batch_size=8, num_defects=14, num_classes=8, **kwargs) -> None:
        super(MixNMatch, self).__init__()
        # prepare net, optimizer and loss
        self.fine_grained_categories = num_defects
        self.super_categories = num_classes
        self.gan_cfg = gan_cfg
        self.bg_loss_wt = bg_loss_wt
        self.batch_size = batch_size

        self.netG, self.netD0, self.netD1, self.netD2, self.BD, self.encoder = load_network(self.gan_cfg, self.fine_grained_categories, self.super_categories)
        self.RF_loss_un = nn.BCELoss(reduction='none')
        self.RF_loss = nn.BCELoss()
        self.CE = CrossEntropy()
        self.L1 = nn.L1Loss()

        # self.avg_param_G = copy_G_params(self.netG)
        
    
        # Other vars
        self.patch_stride = 4.0 
        self.n_out = 24
        self.recp_field = 34

        self.automatic_optimization = False

        

    def _common_step(self, batch, batch_idx, optimizer_idx=None):
        # batch_imgs, *_ = batch

        d_opt, bd_opt, ge_opt = None, None, None
        if self.training:
            opts = self.optimizers()
            bd_opt, ge_opt, *d_opt = opts[0], opts[1], opts[2], None, opts[3]
        # prepare data              
        self.real_img126, self.real_img, self.real_z, self.real_b, self.real_p, self.real_c = self.prepare_epoch_data(batch)
        # forward for both E and G
        self.fake_z, self.fake_b, self.fake_p, self.fake_c = self.encoder( self.real_img, 'softmax' )              
        self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = self.netG( self.real_z, self.real_c, self.real_p, self.real_b, 'code'  )

        # Update Discriminator networks in FineGAN
        
        d_loss_0 = self.train_Dnet(0, d_opt)
        d_loss_2 = self.train_Dnet(2, d_opt)
        # Update Bi Discriminator
        bd_loss = self.train_BD(bd_opt)
        # Update Encoder and G network
        GE_loss = self.train_EG(ge_opt)
        # for avg_p, p in zip( self.avg_param_G, self.netG.parameters() ):
        #     avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        return {f'GE_loss': GE_loss, f'BD_loss': bd_loss, f'd_loss_0': d_loss_0, f'd_loss_2': d_loss_2}
        

    def training_epoch_end(self, outputs):
        GE_avg_loss = torch.stack([x['GE_loss'] for x in outputs]).mean()
        BD_avg_loss = torch.stack([x['BD_loss'] for x in outputs]).mean()
        D0_avg_loss = torch.stack([x['d_loss_0'] for x in outputs]).mean()
        D2_avg_loss = torch.stack([x['d_loss_2'] for x in outputs]).mean()
        loss_dict = {f'train_GE_loss': GE_avg_loss, f'train_BD_loss': BD_avg_loss, f'train_D0_loss': D0_avg_loss, f'train_D2_loss': D2_avg_loss}
        self.log_dict(loss_dict)
        

    def validation_epoch_end(self, outputs):
        GE_avg_loss = torch.stack([x['GE_loss'] for x in outputs]).mean()
        BD_avg_loss = torch.stack([x['BD_loss'] for x in outputs]).mean()
        D0_avg_loss = torch.stack([x['d_loss_0'] for x in outputs]).mean()
        D2_avg_loss = torch.stack([x['d_loss_2'] for x in outputs]).mean()
        loss_dict = {f'val_GE_loss': GE_avg_loss, f'val_BD_loss': BD_avg_loss, f'val_D0_loss': D0_avg_loss, f'val_D2_loss': D2_avg_loss}
        self.log_dict(loss_dict, logger=True)
        self.log('val_loss', GE_avg_loss + D0_avg_loss + D2_avg_loss + BD_avg_loss, sync_dist=True)
        

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss}
        return res


    def configure_optimizers(self):
        optimizersD, optimizerBD, optimizerGE = define_optimizers(self.netG, [self.netD0, self.netD1, self.netD2], self.BD, self.encoder)
        return [optimizerBD, optimizerGE, optimizersD[0], optimizersD[2]]


    def prepare_code(self):
        free_z = torch.FloatTensor( self.batch_size, self.gan_cfg.z_dim ).normal_(0, 1).to(self.device)

        free_c = torch.zeros( self.batch_size, self.fine_grained_categories , device=self.device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, self.fine_grained_categories)
        for i, idx in enumerate(idxs):
            free_c[i,idx] = 1
        free_p = torch.zeros( self.batch_size, self.super_categories , device=self.device)
        idxs = torch.LongTensor( self.batch_size ).random_(0, self.super_categories)
        for i, idx in enumerate(idxs):
            free_p[i,idx] = 1
        free_b = torch.zeros( self.batch_size, self.fine_grained_categories , device=self.device)
        idxs = torch.LongTensor( self.batch_size ).random_( 0, self.fine_grained_categories )
        for i, idx in enumerate(idxs):
            free_b[i,idx] = 1

        return free_z, free_b, free_p, free_c

    

    def prepare_epoch_data(self, data):

        real_img126, real_img, real_c = data 
        # real_img126 = real_img126.to(self.device)
        # real_img = real_img.to(self.device)

        real_p = child_to_parent(real_c, self.fine_grained_categories, self.super_categories)
        real_z = torch.FloatTensor( real_c.shape[0], self.gan_cfg.z_dim ).normal_(0, 1).to(device=self.device) 
        real_c = real_c.to(self.device)
        real_b = real_c             

        return  real_img126, real_img, real_z, real_b, real_p, real_c


    def train_Dnet(self, idx, d_opt=None):

        assert(idx == 0 or idx == 2)
  
        # choose net and opt  
        netD = self.__getattr__(f'netD{idx}')
        if d_opt:
            d_opt[idx].zero_grad()
        # choose real and fake images
        if idx == 0:
            real_img = self.real_img126
            fake_img = self.fake_imgs[0]
        elif idx == 2:
            real_img = self.real_img
            fake_img = self.fake_imgs[2]   
        
        # # # # # # # #for background stage now  # # # # # # #
        if idx == 0:

            # go throung D net to get prediction
            class_prediction, real_prediction = netD(real_img) 
            _, fake_prediction = netD( fake_img.detach() )   

            real_label = torch.ones_like(real_prediction)
            fake_label = torch.zeros_like(fake_prediction)     
            weights_real = torch.ones_like(real_prediction)
            
            # for i in range( self.batch_size ):

            #     x1 = self.warped_bbox[0][i]
            #     x2 = self.warped_bbox[2][i]
            #     y1 = self.warped_bbox[1][i]
            #     y2 = self.warped_bbox[3][i]

            #     a1 = max(torch.tensor(0).float().to(device=self.device), torch.ceil((x1 - self.recp_field)/self.patch_stride))
            #     a2 = min(torch.tensor(self.n_out - 1).float().to(device=self.device), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - x2)/self.patch_stride)) + 1
            #     b1 = max(torch.tensor(0).float().to(device=self.device), torch.ceil( (y1 - self.recp_field)/self.patch_stride))
            #     b2 = min(torch.tensor(self.n_out - 1).float().to(device=self.device), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - y2)/self.patch_stride)) + 1

            #     if (x1 != x2 and y1 != y2):
            #         weights_real[i, :, a1.type(torch.int): a2.type(torch.int), b1.type(torch.int): b2.type(torch.int)] = 0.0

            norm_fact_real = weights_real.sum()
            norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]

            # Real/Fake loss for 'real background' (on patch level)
            real_prediction_loss = self.RF_loss_un( real_prediction, real_label )
            # Masking output units which correspond to receptive fields which lie within the bounding box
            real_prediction_loss = torch.mul(real_prediction_loss, weights_real).mean()
            # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
            if (norm_fact_real > 0):
                real_prediction_loss = real_prediction_loss * ((norm_fact_fake * 1.0) / (norm_fact_real * 1.0))

            # Real/Fake loss for 'fake background' (on patch level)
            fake_prediction_loss = self.RF_loss_un(fake_prediction, fake_label).mean()        
          
            # Background/foreground classification loss
            class_prediction_loss = self.RF_loss_un( class_prediction, weights_real ).mean()  

            # add three losses together 
            D_loss = self.bg_loss_wt*(real_prediction_loss + fake_prediction_loss) + class_prediction_loss
      

        # # # # # # # #for child stage now (only real/fake discriminator)  # # # # # # # 
        if idx == 2:

            # go through D net to get data
            _, real_prediction = netD(real_img) 
            _, fake_prediction = netD( fake_img.detach() )

            # get real/fake lables
            real_label = torch.ones_like(real_prediction)
            fake_label = torch.zeros_like(fake_prediction) 
 
            # get loss 
            real_prediction_loss = self.RF_loss(real_prediction, real_label)         
            fake_prediction_loss = self.RF_loss(fake_prediction, fake_label)
            D_loss = real_prediction_loss+fake_prediction_loss
        if d_opt:
            self.manual_backward(D_loss)
            d_opt[idx].step()

        return D_loss



    def train_BD(self, bd_opt=None):
        if bd_opt:
            bd_opt.zero_grad()

        # make prediction on pairs 
        pred_enc_z, pred_enc_b, pred_enc_p, pred_enc_c = self.BD(  self.real_img, self.fake_z.detach(), self.fake_b.detach(), self.fake_p.detach(), self.fake_c.detach() )
        pred_gen_z, pred_gen_b, pred_gen_p, pred_gen_c = self.BD(  self.fake_imgs[2].detach(), self.real_z, self.real_b, self.real_p, self.real_c )
       
        real_data = [ self.real_img, self.fake_z.detach(), self.fake_b.detach(), self.fake_p.detach(), self.fake_c.detach() ]
        fake_data = [ self.fake_imgs[2].detach(), self.real_z, self.real_b, self.real_p, self.real_c ]
        penalty = cal_gradient_penalty( self.BD, real_data, fake_data, self.device, type='mixed', constant=1.0)

        D_loss =  -( pred_enc_z.mean()+pred_enc_b.mean()+pred_enc_p.mean()+pred_enc_c.mean()  ) + ( pred_gen_z.mean()+pred_gen_b.mean()+pred_gen_p.mean()+pred_gen_c.mean() ) + penalty*10
        if bd_opt:
            self.manual_backward(D_loss)
            bd_opt.step()
        return D_loss
      


    def train_EG(self, ge_opt=None):

        if ge_opt:
            ge_opt.zero_grad()

        # reconstruct code and calculate loss 
        self.rec_p, _ = self.netD1( self.fg_mk[0])
        self.rec_c, _ = self.netD2( self.fg_mk[1])
        p_code_loss = self.CE( self.rec_p , self.real_p )
        c_code_loss = self.CE( self.rec_c,  self.real_c )

        # pred code and calculate loss (here no code constrain)
        free_z, free_b, free_p, free_c = self.prepare_code()
        with torch.no_grad():
            free_fake_imgs, _, _, _ = self.netG( free_z, free_c, free_p, free_b, 'code'  )                   
        pred_z, pred_b, pred_p, pred_c = self.encoder( free_fake_imgs[2].detach(),   'logits' )
        z_pred_loss = self.L1( pred_z , free_z )
        b_pred_loss = self.CE( pred_b , free_b )
        p_pred_loss = self.CE( pred_p , free_p )
        c_pred_loss = self.CE( pred_c,  free_c )        
    
    
        # aux and backgroud real/fake loss
        self.bg_class_pred, self.bg_rf_pred = self.netD0( self.fake_imgs[0] ) 
        bg_rf_loss = self.RF_loss( self.bg_rf_pred, torch.ones_like( self.bg_rf_pred ) )* self.bg_loss_wt
        bg_class_loss = self.RF_loss( self.bg_class_pred, torch.ones_like( self.bg_class_pred ) )

        # child image real/fake loss  
        _, self.child_rf_pred = self.netD2( self.fake_imgs[-1] )
        child_rf_loss = self.RF_loss( self.child_rf_pred, torch.ones_like(self.child_rf_pred) )
  
        # fool BD loss
        pred_enc_z, pred_enc_b, pred_enc_p, pred_enc_c = self.BD(  self.real_img, self.fake_z, self.fake_b, self.fake_p, self.fake_c )
        pred_gen_z, pred_gen_b, pred_gen_p, pred_gen_c = self.BD(  self.fake_imgs[2], self.real_z, self.real_b, self.real_p, self.real_c )
        fool_BD_loss = ( pred_enc_z.mean()+pred_enc_b.mean()+pred_enc_p.mean()+pred_enc_c.mean()  ) - ( pred_gen_z.mean()+pred_gen_b.mean()+pred_gen_p.mean()+pred_gen_c.mean() ) 
             
        EG_loss =  (p_code_loss+c_code_loss) + (bg_rf_loss+bg_class_loss) + child_rf_loss + fool_BD_loss + (5*z_pred_loss+5*b_pred_loss+10*p_pred_loss+10*c_pred_loss)
        
        if ge_opt:
            self.manual_backward(EG_loss)
            ge_opt.step()

        return EG_loss
