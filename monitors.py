import os
import numpy as np
from mxboard import SummaryWriter
from mxnet import nd
import mxnet as mx

# MxBoard / Tensorboard Monitoring Setup

class Monitoring:
    def __init__(self, num_workers, path, now, context):
        self.update = []
        self.rewards = []
        self.runner_rewards= [[] for i in range(num_workers)]
        self.runner_episodes = [0 for i in range(num_workers)]
        self.runner_cum_rewards = [0 for i in range(num_workers)]
        self.runner_mean100ep = [[] for i in range(num_workers)]
        self.mean100 = []
        self.max_reward = []
        self.runner_episode_lengths = [[] for i in range(num_workers)]

        self.entropy_loss = nd.zeros((1), ctx=context) 
        self.actor_loss =nd.zeros((1), ctx=context) 
        self.critic_loss =nd.zeros((1), ctx=context) 
        self.ppo_loss =nd.zeros((1), ctx=context) 
        self.stim_loss =nd.zeros((1), ctx=context) 
        self.recon_loss =nd.zeros((1), ctx=context) 
        self.sparsity_loss =nd.zeros((1), ctx=context) 
        self.total_loss =nd.zeros((1), ctx=context) 
        
        
        self.min_action_prob = nd.zeros((1), ctx=context) 
        self.max_action_prob = nd.zeros((1), ctx=context) 
        self.avg_action_prob = nd.zeros((1), ctx=context) 
        self.std_action_prob = nd.zeros((1), ctx=context) 
        self.avg_value = nd.zeros((1), ctx=context) 
        self.min_value = nd.zeros((1), ctx=context) 
        self.max_value = nd.zeros((1), ctx=context) 
        self.std_value = nd.zeros((1), ctx=context) 
        self.value_bias_grad = []
        self.action_bias_grad = []
        self.avg_grad = []

        self.ratio = nd.zeros((1), ctx=context) 
        self.entropy_loss_buffer = nd.zeros((1), ctx=context) 
        self.actor_loss_buffer =nd.zeros((1), ctx=context) 
        self.critic_loss_buffer =nd.zeros((1), ctx=context)
        self.ppo_loss_buffer =nd.zeros((1), ctx=context)
        self.stim_loss_buffer =nd.zeros((1), ctx=context) 
        self.recon_loss_buffer =nd.zeros((1), ctx=context) 
        self.sparsity_loss_buffer =nd.zeros((1), ctx=context) 
        self.total_loss_buffer =nd.zeros((1), ctx=context) 
        self.ratio_buffer =nd.zeros((1), ctx=context) 
        
        self.eval_rewards = []
        self.eval_nsteps =[]
        
        self.eval_rewards_avg = []
        self.eval_nsteps_avg =[]
        self.eval_best_rewards=[]
        self.eval_rewards_std=[]
        self.eval_nsteps_std = []
        
        self.now=now
        
        # self.sw = SummaryWriter(logdir= path + '/logs_'+now, flush_secs=5)


    def process_episode(self, rewards, ep, runner, eps_length, path, update):
        self.rewards.append(rewards)

        if len(self.rewards)>100:
            self.mean100.append(np.mean(self.rewards[-100:]))
        else:
            self.mean100.append(np.mean(self.rewards))
        self.max_reward.append(np.max(self.rewards))
        
        self.runner_rewards[runner].append(rewards)
        self.runner_cum_rewards[runner]+= rewards
        self.runner_episodes[runner]+= 1
        self.runner_episode_lengths[runner].append(eps_length)
        
        if len(self.runner_rewards[runner])>100:
            self.runner_mean100ep[runner].append(np.mean(self.runner_rewards[runner][-100:]))
        else:
            self.runner_mean100ep[runner].append(np.mean(self.runner_rewards[runner]))

    def save_np(self,update,path, game):
        
        np.save(path+'npdict_e2e_{}_{}_update{}.npy'.format(game, self.now, update), 
                   {'Rewards/Rewards':self.rewards,
                     'Rewards/Avg._Reward_Last_100':self.mean100,
                     'Rewards/Max._Reward':self.max_reward,
                     'Rewards/Runner_Reward': self.runner_rewards, 
                     'Rewards/Runner_Avg_Reward_Last_100_Ep': self.runner_mean100ep, 
                     'Episode_Length': self.runner_episode_lengths
                   })
        

        
    def process_rollout(self, data):
        
        self.min_action_prob=nd.concat(self.min_action_prob, data['action_dists'].min(), dim=0)
        self.max_action_prob=nd.concat(self.max_action_prob, data['action_dists'].max(), dim=0)
        mean_a, var_a=nd.moments(data=data['action_dists'],axes=[0])
        self.avg_action_prob=nd.concat(self.avg_action_prob, mean_a, dim=0)
        self.std_action_prob=nd.concat(self.std_action_prob, nd.sqrt(var_a), dim=0)

        self.min_value=nd.concat(self.min_value, data['values'].min(), dim=0)
        self.max_value=nd.concat(self.max_value, data['values'].max(), dim=0)
        mean_v, var_v=nd.moments(data=data['values'],axes=[0])
        self.avg_value=nd.concat(self.avg_value, mean_v, dim=0)
        self.std_value=nd.concat(self.std_value, nd.sqrt(var_v), dim=0)        
    

    def process_minibatch_loss(self, entropy_loss, actor_loss, critic_loss, ppo_loss, stim_loss, recon_loss, sparsity_loss, total_loss, ratio):

        self.entropy_loss_buffer=nd.concat(self.entropy_loss_buffer,entropy_loss,dim=0)
        self.actor_loss_buffer=nd.concat(self.actor_loss_buffer,actor_loss,dim=0) 
        self.critic_loss_buffer=nd.concat(self.critic_loss_buffer,critic_loss,dim=0) 
        self.ppo_loss_buffer=nd.concat(self.ppo_loss_buffer,ppo_loss,dim=0) 
        self.stim_loss_buffer=nd.concat(self.stim_loss_buffer,stim_loss,dim=0) 
        self.recon_loss_buffer=nd.concat(self.recon_loss_buffer,recon_loss,dim=0) 
        self.sparsity_loss_buffer=nd.concat(self.sparsity_loss_buffer,sparsity_loss,dim=0) 
        self.total_loss_buffer=nd.concat(self.total_loss_buffer,total_loss,dim=0) 
        self.ratio_buffer=nd.concat(self.ratio_buffer,ratio,dim=0) 

        
    def process_grads(self, update, grads):
            
        self.value_bias_grad.append(grads[-1]) 
        self.action_bias_grad.append(grads[-3][0]) 


    def process_update(self, update, context):
        self.update.append(update)
        
        self.entropy_loss=nd.concat(self.entropy_loss,self.entropy_loss_buffer[1:].mean(),dim=0)
        self.actor_loss=nd.concat(self.actor_loss,self.actor_loss_buffer[1:].mean(),dim=0) 
        self.critic_loss=nd.concat(self.critic_loss,self.critic_loss_buffer[1:].mean(),dim=0) 
        self.ppo_loss=nd.concat(self.ppo_loss,self.ppo_loss_buffer[1:].mean(),dim=0) 
        self.stim_loss=nd.concat(self.stim_loss,self.stim_loss_buffer[1:].mean(),dim=0) 
        self.recon_loss=nd.concat(self.recon_loss,self.recon_loss_buffer[1:].mean(),dim=0)
        self.sparsity_loss=nd.concat(self.sparsity_loss,self.sparsity_loss_buffer[1:].mean(),dim=0) 
        self.total_loss=nd.concat(self.total_loss,self.total_loss_buffer[1:].mean(),dim=0) 
        self.ratio=nd.concat(self.ratio,self.ratio_buffer[1:].mean(),dim=0) 
    
        
        self.entropy_loss_buffer = nd.zeros((1), ctx=context) #[]
        self.actor_loss_buffer =nd.zeros((1), ctx=context) #[]
        self.critic_loss_buffer =nd.zeros((1), ctx=context) # []
        self.ppo_loss_buffer =nd.zeros((1), ctx=context) # []
        self.stim_loss_buffer =nd.zeros((1), ctx=context) # []
        self.recon_loss_buffer =nd.zeros((1), ctx=context) # []
        self.sparsity_loss_buffer =nd.zeros((1), ctx=context) # []
        self.total_loss_buffer =nd.zeros((1), ctx=context) # []
        self.ratio_buffer =nd.zeros((1), ctx=context) # []

    def save_nd(self,update,path,game):
        nd.save(path+'datadict_e2e_{}_{}_update{}.json'.format(game,self.now, update),
                {'Losses/Critic_Loss':self.critic_loss[1:],
                 'Losses/Actor_Loss':self.actor_loss[1:],
                 'Losses/Entropy_Loss':self.entropy_loss[1:],
                 'Losses/PPO_Loss':self.ppo_loss[1:],
                 'Losses/Stimulation_Loss':self.stim_loss[1:],
                 'Losses/Reconstruction_Loss':self.recon_loss[1:],
                 'Losses/Sparsity_Loss':self.sparsity_loss[1:],
                 'Losses/Total_Loss':self.total_loss[1:],
                 'Probabilities/Average_Action_Probability': self.avg_action_prob[1:],
                 'Probabilities/Min_Action_Probability': self.min_action_prob[1:],
                 'Probabilities/Max_Action_Probability': self.max_action_prob[1:],
                 'Probabilities/Std_Action_Probability': self.std_action_prob[1:],
                 'Values/Average_State_Value':self.avg_value[1:],
                 'Values/Min_State_Value':self.min_value[1:],
                 'Values/Max_State_Value':self.max_value[1:],
                 'Values/Std_State_Value':self.std_value[1:],
                 'Ratio/Average_Ratio':self.ratio[1:]
              })


    def update_mxboard(self, path, now):
        self.sw.add_scalar(tag='Losses/Critic_Loss',                           value=self.critic_loss[-1].mean(),         global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Actor_Loss',                            value=self.actor_loss[-1].mean(),          global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Entropy_Loss',                          value=self.entropy_loss[-1].mean(),        global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/PPO_Loss',                              value=self.ppo_loss[-1].mean(),             global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Stimulation_Loss',                      value=self.stim_loss[-1].mean(),            global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Reconstruction_Loss',                   value=self.recon_loss[-1].mean(),           global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Sparsity_Loss',                         value=self.sparsity_loss[-1].mean(),        global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Total_Loss',                            value=self.total_loss[-1].mean(),            global_step=self.update[-1])
        self.sw.add_scalar(tag='Probabilities/Average_Action_Probability',     value=self.avg_action_prob[-1],     global_step=self.update[-1])
        self.sw.add_scalar(tag='Probabilities/Min._Action_Probability',        value=self.min_action_prob[-1],     global_step=self.update[-1])
        self.sw.add_scalar(tag='Probabilities/Max._Action_Probability',        value=self.max_action_prob[-1],     global_step=self.update[-1])
        self.sw.add_scalar(tag='Probabilities/Std._Dev._Action_Probability',   value=self.std_action_prob[-1],     global_step=self.update[-1])
        self.sw.add_scalar(tag='Values/Average_State_Value',                   value=self.avg_value[-1],           global_step=self.update[-1])
        self.sw.add_scalar(tag='Values/Min._State_Value',                      value=self.min_value[-1],           global_step=self.update[-1])
        self.sw.add_scalar(tag='Values/Max._State_Value',                      value=self.max_value[-1],           global_step=self.update[-1])
        self.sw.add_scalar(tag='Values/Std._Dev._State_Value',                value=self.std_value[-1],           global_step=self.update[-1])
        self.sw.add_scalar(tag='Gradients/Final_Value_Bias_Gradient',         value=self.value_bias_grad[-1],     global_step=self.update[-1])
        self.sw.add_scalar(tag='Gradients/Average_Final_Action_Bias_Gradient', value=self.action_bias_grad[-1],    global_step=self.update[-1])
        self.sw.add_scalar(tag='Ratio/Average_Ratio',                          value=self.ratio[-1],               global_step=self.update[-1])
        
        
        if np.min(self.runner_episodes)>0:
            min_runner_eps_count=np.min(self.runner_episodes)
            runner_rewards_from_same_eps=[runner[min_runner_eps_count-1] for runner in self.runner_rewards]
            mean_eps_reward=np.mean(runner_rewards_from_same_eps)
            
            self.sw.add_scalar(tag='Rewards/Avg_Runners_Reward_per_Eps',           value=int(mean_eps_reward),          global_step=int(min_runner_eps_count))

        self.sw.add_scalar(tag='Losses/Critic_Loss_last100avg',                           value=np.mean(self.critic_loss[-100:]),         global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Actor_Loss_last100avg',                            value=np.mean(self.actor_loss[-100:]),          global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Entropy_Loss_last100avg',                          value=np.mean(self.entropy_loss[-100:]),        global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/PPO_Loss_last100avg',                              value=np.mean(self.ppo_loss[-100:]),              global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Stimulation_Loss_last100avg',                      value=np.mean(self.stim_loss[-100:]),             global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Reconstruction_Loss_last100avg',                   value=np.mean(self.recon_loss[-100:]),            global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Sparsity_Loss_last100avg',                          value=np.mean(self.sparsity_loss[-100:]),        global_step=self.update[-1])
        self.sw.add_scalar(tag='Losses/Total_Loss_last100avg',                            value=np.mean(self.total_loss[-100:]),            global_step=self.update[-1])

        os.makedirs('./exports', exist_ok=True)
        self.sw.export_scalars(path+'scalars_' +now+'.json')