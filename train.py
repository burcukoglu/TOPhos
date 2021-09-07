import os
import gym
import numpy as np
import cv2
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn, rnn, loss
from mxnet.gluon.nn import HybridSequential, Sequential #LeakyReLU
import mxnet as mx
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
#import pandas_bokeh
#import gc
import multiprocessing 

import multiprocessing.connection
import datetime
import math
import time
import imageio
#from skimage.transform import resize
import noise

import models 
import utils
import multiprocess 
import monitors


# Configuration
game            = "BreakoutDeterministic-v4"
env             = gym.make(game)
seq_size        = 4
context         = mx.gpu(0) #mx.cpu()
entropy_coeff   = .02 
critic_coeff    = .5
num_workers     = 8 #16 #1 
batch_steps     = 128
w, h            = 128,128 
w_enc, h_enc    = 32, 32
w_s, h_s        = 84, 84
w_dec, h_dec    = 256, 256  # 128, 128 #256, 256
gamma           = .99
lamda           = .95
updates         = 15000 
epochs          = 4
batch_steps     = 128 
n_mini_batch    = 4 #ideally 4, but may need to decrease until 64 by doubling the value (sometimes to 128 or 256 even)
batch_size      = num_workers * batch_steps
mini_batch_size = batch_size // n_mini_batch
states          = nd.zeros((num_workers, 4, h_dec, w_dec), dtype=np.float32, ctx=context)

states_new      = np.zeros((num_workers, 210, 160, 3), dtype=np.float32)

states_pre_stacked          = np.zeros((num_workers, seq_size, h, w), dtype=np.float32)

weight_loss_ppo = .3 
kappa=  0 

all_grads       = []
learning_rate   = 2.5e-4
initial_learning_rate = learning_rate

#  states_list_cur_eps = [] 
# states_phos_list_cur_eps = []
# states_pre_list_cur_eps = []
# states_enc_list_cur_eps = []
# states_dec_list_cur_eps = []

# gif_pre_states =[[] for i in range(num_workers)]
# gif_enc_states =[[] for i in range(num_workers)]
# gif_phos_states =[[] for i in range(num_workers)]
# gif_dec_states =[[] for i in range(num_workers)]

# states_pre          = np.zeros((num_workers, h, w), dtype=np.float32)
# states_enc          = np.zeros((num_workers, h_enc, w_enc), dtype=np.float32)
# states_dec          = np.zeros((num_workers, h, w), dtype=np.float32)
# states_phos         = np.zeros((num_workers, h_dec, w_dec), dtype=np.float32)


START = True #True if starting training from the beginning, False if continuing training from a time point

if START:
    cur_eps         = np.zeros((num_workers), dtype=np.int32) #current episode rewards
    cur_eps_len     = np.zeros((num_workers), dtype=np.int32) #current episode length
    total_episodes  = 0
    lives           = np.zeros(num_workers, dtype=np.int32) + env.unwrapped.ale.lives()
    start_update    = 0

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print('now',now)
    now_restart=now
    print('now_restart', now_restart)
    PATH='/Users/u485148/lasts/encsimdec/ee_'+game.split('D')[0]+'_'+now+'_wp'+str(weight_loss_ppo)+'_k'+str(kappa)+'/'
    # PATH='/huge/burkuc/atari/enc_sim_dec/ee_'+game.split('D')[0]+'_'+now+'_wp'+str(weight_loss_ppo)+'_k'+str(kappa)+'/'
    print(PATH)
    os.makedirs(PATH, exist_ok=True)

else:
    params_file = 'e2eBreakout_params_2021-09-03--12-23-40_update2_wp0.3_k0'
    now = params_file.split('_')[-4] #'' #
    print('now',now)
    now_restart=datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") 
    print('now_restart', now_restart)
    #   PATH='/content/gdrive/My Drive/ppo/enc_sim_dec/'+now+'_wp'+str(weight_loss_ppo)+'_k'+str(kappa)+'/'
    PATH='/Users/u485148/lasts/encsimdec/ee_'+game.split('D')[0]+'_'+now+'_wp'+str(weight_loss_ppo)+'_k'+str(kappa)+'/'
    # PATH='/huge/burkuc/atari/enc_sim_dec/ee_'+game.split('D')[0]+'_'+now+'_wp'+str(weight_loss_ppo)+'_k'+str(kappa)+'/'
    os.makedirs(PATH, exist_ok=True)

    optimizer_states_file = 'optimizerstates_2021-09-03--12-23-40_update2_wp0.3_k0'
    total_episodes_file = 'total_episodes_update2_2021-09-03--12-23-40_2021-09-03--12-31-39.npy'

    cur_eps         = np.load(PATH+'cur_eps_{}.npy'.format(params_file.split('_')[-3])) 
    cur_eps_len     = np.load(PATH+'cur_eps_len_{}.npy'.format(params_file.split('_')[-3])) 
    total_episodes  = np.load(PATH+total_episodes_file) 
    lives           = np.load(PATH+'lives_{}.npy'.format(params_file.split('_')[-3]))
    start_update    = int(params_file.split('_')[-3].split('e')[-1]) +1 


print('current episode rewards',cur_eps)
print('current episode length',cur_eps_len)
print('total_episodes',total_episodes)
print('lives',lives)
print('starting update', start_update)


class CustomLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0): 

        """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss) 
        or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
        stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
        """
        
        # Reconstruction loss
        if recon_loss_type == 'mse':
            self.recon_loss = loss.L2Loss() 
            self.target = 'image'
        # elif recon_loss_type == 'vgg':
        #     self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
        #     self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
        #     self.target = 'image'
        # elif recon_loss_type == 'boundary':
        #     loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
        #     self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
        #     self.target = 'label'

        # Stimulation loss 
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: nd.mean(.5*(x+1)) #torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: nd.mean((.5*(x+1))**2)  #torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0
        
    def __call__(self,image,stimulation,phosphenes,reconstruction,validation=False):    

        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            target = image
        # elif self.target == 'label':
        #     target = label
        
        # Calculate loss
        loss_stimu = self.stimu_loss(stimulation.as_in_context(context)) if self.stimu_loss is not None else nd.zeros(1,ctx=context)

        loss_recon = self.recon_loss(reconstruction.as_in_context(context),target)

        loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
        return loss_total, nd.mean(loss_recon), loss_stimu

#preprocessing for normal vision
# def preprocess(state): 
#     state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (h, w), interpolation=cv2.INTER_AREA)
#     return state/255

#grasycale conversion and resizing before inputting to encoder
def preprocess_before_encoder(state):  
    state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (h, w), interpolation=cv2.INTER_AREA) 
    return state / 255

#processing of frames through encoder and phosphene simulator before inputting to RL agent
def process_with_net(net, pre_state,pmask):
    encoded=net.enc(pre_state)
    phosphenes=net.phosim(encoded, pmask)
    return phosphenes

def summarize(net, context):
    state = env.reset()
    state=preprocess_before_encoder(state)
    state_pre = state
    state = process_with_net(net, state_pre, pmask)
    state = np.array([state.asnumpy(),]*seq_size).reshape((1,-1,h_dec,w_dec))
    net.summary(nd.array(state, ctx=context), nd.array(state_pre, ctx=context))

def reset_state(runner, net):
    runner.child.send(("reset", None))
    state = runner.child.recv()
    state = preprocess_before_encoder(state)
    states_pre_stacked = np.array([state,]*seq_size).reshape((-1,seq_size, h, w))
    state=process_with_net(net, nd.array(state.reshape(-1,1,h,w), ctx=context),pmask)
    state=nd.tile(state, (1, seq_size,1,1)).reshape(seq_size,h_dec,w_dec)
    return state,  states_pre_stacked 

def process_states(pmask,states_new, states,states_pre_stacked, dones, infos, runners, game, lives,net): #, update, step):
    global cur_eps
    global total_episodes
    global monitor
    global cur_eps_len

    states_new = np.stack(states_new) #from (8,) to (8, 210, 160, 3)
    states_new = np.array([preprocess_before_encoder(state) for state in states_new])
    states_pre = states_new.reshape(-1,h,w)
    states_pre_stacked = np.append(states_pre_stacked, states_pre.reshape(-1,1,h,w), axis=1) #(8, 5, 84, 84)
    states_pre_stacked = np.delete(states_pre_stacked, 0, axis=1)  #(8, 4, 84, 84)
    states_new=process_with_net(net, nd.array(states_new.reshape(-1,1,h,w), ctx=context),pmask)
    states = nd.concat(states, states_new, dim=1)
    states = states[:,1:,:,:]

    for idx, [cur_done, runner, cur_info] in enumerate(zip(dones, runners, infos)):
        if cur_done or (cur_info['ale.lives'] < lives[idx] and 'Breakout' in game):
            lives[idx] = cur_info['ale.lives']
            if cur_done:
                monitor.process_episode(cur_eps[idx], total_episodes, idx, cur_eps_len[idx], PATH, update)
                total_episodes += 1
                cur_eps[idx] = 0
                cur_eps_len[idx] = 0
                lives[idx] = env.unwrapped.ale.lives()
                states[idx], states_pre_stacked[idx]  = reset_state(runner, net) #decoder
            dones[idx] = True
    return states, dones, states_pre_stacked

def calculate_advantages( done, rewards, values, states):
    advantages = nd.zeros((num_workers, batch_steps), dtype=np.float32, ctx=context)

    last_advantage = 0
    _, last_value = mainnet.model(states)

    last_value = last_value.reshape(-1) 

    for t in reversed(range(batch_steps)):

        mask = nd.array(1.0 - done[:, t], ctx=context)

        last_value = last_value.as_in_context(context) * mask
        last_advantage = last_advantage * mask
        delta = nd.array(rewards[:, t], ctx=context) + gamma * last_value - values[:, t].as_in_context(context)

        last_advantage = delta + gamma * lamda * last_advantage

        advantages[:, t] = last_advantage

        last_value = values[:, t]

    return advantages


# Batch rollout

def rollout():
    global states
    global states_new
    global states_pre_stacked
    global cur_eps
    global lives
    global total_episodes
    global cur_eps_len
    # global states_list_cur_eps
    # global states_pre_list_cur_eps
    # global states_enc_list_cur_eps
    # global states_dec_list_cur_eps
    # global states_phos_list_cur_eps

    # global states_pre
    # global states_enc
    # global states_dec
    # global states_phos
    
    # Initialize batch
    data = {'rewards': np.zeros((num_workers, batch_steps)), 'values': nd.zeros((num_workers, batch_steps)),
            'log_probs': nd.zeros((num_workers, batch_steps)), 'action_dists': nd.zeros((num_workers, batch_steps, env.action_space.n)),
            'done': np.zeros((num_workers, batch_steps)), 'states': nd.zeros((num_workers, batch_steps, seq_size, h_dec, w_dec)), 
            'actions': np.zeros((num_workers, batch_steps), dtype=np.int32),
            'states_pre_stacked': np.zeros((num_workers, batch_steps, seq_size, h, w)) }
            # 'states_pre': np.zeros((num_workers, batch_steps, h, w)),
            # 'states_enc': np.zeros((num_workers, batch_steps, h_enc, w_enc)), 'states_dec': np.zeros((num_workers, batch_steps, h, w)),
            # 'states_phos': np.zeros((num_workers, batch_steps, h_dec, w_dec))}


    for step in range(batch_steps):

        #probs, v = mainnet.model(nd.array(states, ctx=context))
        probs, v = mainnet.model(states)
        act = mx.nd.sample_multinomial(probs)  
        act_probs=nd.pick(probs,act)
        log_probs=nd.log(act_probs+1e-10) 
        data['states'][:, step] = states
        data['action_dists'][:, step] = probs # .asnumpy()
        data['values'][:, step] = v.reshape(-1) #.asnumpy()
        data['actions'][:, step] = act.reshape(-1).asnumpy()
        data['log_probs'][:, step] = log_probs.reshape(-1) #.asnumpy()
        data['states_pre_stacked'][:, step] = states_pre_stacked

        # data['states_pre'][:, step] = states_pre
        # data['states_enc'][:, step] = states_enc
        # data['states_dec'][:, step] = states_dec
        # data['states_phos'][:, step] = states_phos

        for idx, runner in enumerate(runners):
            runner.child.send(("step", data['actions'][idx,step]))
        states_new, data['rewards'][:, step], data['done'][:, step], info = np.transpose(np.array([runner.child.recv() for runner in runners], dtype=object))
        cur_eps[:] = cur_eps + data['rewards'][:, step]
        cur_eps_len[:] = cur_eps_len + np.ones((num_workers), dtype=np.int32) #npc
        # states_list_cur_eps.append(np.stack(states_new))
        
        # #generate gifs for ending episodes
        # for idx, cur_done in enumerate(data['done'][:, step]):
        #     if cur_done:
        #         gif_states= [item[idx] for item in states_list_cur_eps] 
        #         gif_pre_states = [item[idx].reshape(h,w,-1) for item in states_pre_list_cur_eps] 
        #         gif_enc_states= [item[idx].reshape(h_enc,w_enc,-1) for item in states_enc_list_cur_eps] 
        #         gif_phosphene_states = [item[idx].reshape(h_dec,w_dec,-1) for item in states_phos_list_cur_eps]
        #         gif_dec_states=  [item[idx].reshape(h,w,-1) for item in states_dec_list_cur_eps] 

        #         utils.generate_gif(update,gif_states, cur_eps[idx],PATH, step, idx)
        #         utils.generate_gif_pre(update,gif_pre_states, cur_eps[idx],PATH, step, idx)
        #         utils.generate_gif_enc(update,gif_enc_states, cur_eps[idx],PATH, step, idx)
        #         utils.generate_gif_phos(update,gif_phosphene_states, cur_eps[idx],PATH, step, idx)
        #         utils.generate_gif_dec(update,gif_dec_states, cur_eps[idx],PATH, step, idx)

        states, data['done'][:, step],  states_pre_stacked = process_states(pmask,states_new, states,states_pre_stacked, data['done'][:, step], info, runners, game, lives, mainnet) 

        # states_pre_list_cur_eps.append(states_pre)
        # states_enc_list_cur_eps.append(states_enc)
        # states_phos_list_cur_eps.append(states_phos)
        # states_dec_list_cur_eps.append(states_dec)
        

    data['advantages'] = calculate_advantages(data['done'], data['rewards'], data['values'], states)

    for key, val in data.items():
        val = val.reshape(val.shape[0] * val.shape[1], *val.shape[2:])
        if isinstance(val, mx.ndarray.NDArray):
            data[key] = val.as_in_context(context)
        else:
            data[key] = nd.array(val, ctx=context)

    monitor.process_rollout(data)

    return data

def get_pMask(size=(h_dec,w_dec),phosphene_density=32,seed=1,jitter_amplitude=0,dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = nd.zeros(size)


    # Custom 'dropout_map'
    p_dropout = utils.perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)

        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density, jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)

        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,1.], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = 1.
        
            
    return pMask       


monitor = monitors.Monitoring(num_workers, PATH, now, context)

pmask = get_pMask(jitter_amplitude=0,dropout=False).as_in_context(context)

mainnet = models.MainModel(env.action_space.n, context) #(pmask,env.action_space.n, context)

if START:
    mainnet.initialize(ctx=context)
else:
    mainnet.load_parameters(PATH+params_file,ctx=context)

# mainnet.hybridize()
    
optimizer = gluon.Trainer(mainnet.collect_params(), 'Adam', {'learning_rate': 2.5e-4})

if not START:
    optimizer.load_states(PATH+optimizer_states_file)

loss_function = CustomLoss(recon_loss_type='mse',
                                            recon_loss_param=0,
                                            stimu_loss_type=None,
                                            kappa=kappa) 
                                       

runners = [multiprocess.Runner(game) for i in range(num_workers)]


for i, runner in enumerate(runners):
    states[i],  states_pre_stacked[i] = reset_state(runner, mainnet)

# Main loop
for update in range(start_update,updates):
    print('update', update)
    progress = update / updates
    learning_rate = initial_learning_rate * (1 - progress)
    clip_range = 0.10 

    samples = rollout()
    for _ in range(epochs):

        indexes = nd.array(range(batch_size), ctx=context).astype(np.int32)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for key, val in samples.items():
                mini_batch[key] = val[mini_batch_indexes]

            with autograd.record():
                batch_return = mini_batch['values']+ mini_batch['advantages']
                advantages = utils.normalize(mini_batch['advantages'])
                probs, value, mini_batch_states_enc, mini_batch_phosphene, mini_batch_states_dec = mainnet(pmask,mini_batch['states_pre_stacked'][:,0,:,:], mini_batch['states_pre_stacked'][:,1,:,:],mini_batch['states_pre_stacked'][:,2,:,:],mini_batch['states_pre_stacked'][:,3,:,:])
#                
      

                actor_loss, ratio = utils.calc_actor_loss(probs, mini_batch, advantages, clip_range)
                critic_loss = utils.calc_critic_loss(mini_batch, value, clip_range, batch_return)
                entropy_loss = utils.calc_entropy_loss(probs)

                loss_ppo = actor_loss + critic_coeff * critic_loss - entropy_coeff * entropy_loss


                loss_stim, loss_recon, loss_sparsity = loss_function(image=mini_batch['states_pre_stacked'][:,-1,:,:],
                                 stimulation=mini_batch_states_enc[:,-1,:,:],
                                 phosphenes= mini_batch_phosphene[:,-1,:,:],     #mini_batch['states'][:,-1,:,:], #not used in loss 
                                 reconstruction=mini_batch_states_dec[:,-1,:,:])
                

                loss = weight_loss_ppo * loss_ppo  + (1-weight_loss_ppo) * nd.mean(loss_stim)


            monitor.process_minibatch_loss(entropy_coeff*entropy_loss, actor_loss, critic_loss,loss_ppo, loss_stim, loss_recon, loss_sparsity, loss,  nd.mean(ratio))

            #to change learning rate during training
            optimizer.set_learning_rate(learning_rate)

            loss.backward()
            
            grads = [i.grad(context) for i in mainnet.collect_params().values() if i._grad is not None]
            gluon.utils.clip_global_norm(grads, 0.5, check_isfinite=False) #addedfalsetoavoidnumpy
#            
            optimizer.step(1) #,ignore_stale_grad=True)

    monitor.process_grads(update, [i.grad(context) for i in mainnet.collect_params().values() if i._grad is not None])
    monitor.process_update(update, context)


    if (update)%100==0:
        mainnet.save_parameters(PATH + 'e2e'+game.split('D')[0]+"_params_"  + now + "_update" + str(update) +'_wp'+str(weight_loss_ppo)+'_k'+str(kappa))
        optimizer.save_states(PATH +'optimizerstates_'+now + "_update" + str(update) +'_wp'+str(weight_loss_ppo)+'_k'+str(kappa))
        np.save(PATH+'total_episodes_update{}_{}_{}.npy'.format(update, now_restart, datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")) ,total_episodes) #to keep track of running time, especially useful after every restart to continue training
        np.save(PATH+'cur_eps_update{}.npy'.format(update) ,cur_eps)
        np.save(PATH+'cur_eps_len_update{}.npy'.format(update) ,cur_eps_len)
        np.save(PATH+'lives_update{}.npy'.format(update) ,lives)
        monitor.save_nd(update, PATH, game.split('D')[0])
        monitor.save_np(update, PATH, game.split('D')[0])

mainnet.save_parameters(PATH + 'e2e'+game.split('D')[0]+"_params_" + now + "_update" + str(update) +'_wp'+str(weight_loss_ppo)+'_k'+str(kappa))
optimizer.save_states(PATH +'optimizerstates_'+now + "_update" + str(update) +'_wp'+str(weight_loss_ppo)+'_k'+str(kappa))
np.save(PATH+'total_episodes_update{}_{}_{}.npy'.format(update, now_restart, datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")) ,total_episodes)
np.save(PATH+'cur_eps_update{}.npy'.format(update) ,cur_eps)
np.save(PATH+'cur_eps_len_update{}.npy'.format(update) ,cur_eps_len)
np.save(PATH+'lives_update{}.npy'.format(update) ,lives)
monitor.save_nd(update, PATH,game.split('D')[0])
monitor.save_np(update, PATH,game.split('D')[0])

