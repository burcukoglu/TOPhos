import cv2
import numpy as np
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn, rnn, loss
import mxnet as mx
import imageio
import noise
#from skimage.transform import resize

w_dec, h_dec = 256,256 

#advantage normalization
def normalize(adv):
    return (adv - adv.mean()) / (nd.sqrt(nd.power(adv-adv.mean(),2).sum() / len(adv)) + 1e-8)


def calc_actor_loss(probs, mini_batch, advantages, clip_range):
    
    log_probs = nd.log(nd.pick(probs,mini_batch['actions'])+1e-10)
    
    #log ratio of how much the policy changed: new policy/old_policy
    ratio = nd.exp(log_probs - mini_batch['log_probs'])

    #clipping ratio for signaling of how much change in policy we are willing to tolerate
    clipped_ratio = nd.clip(ratio,1.0 - clip_range,1.0 + clip_range)
    
    #take the min between ratio and clipped ratio
    actor_loss = nd.concat((ratio * advantages).reshape(1,-1),(clipped_ratio * advantages).reshape(1,-1), dim = 0)
    actor_loss = nd.min(actor_loss, axis=0)

    #take negative of the positive value for final loss calculation
    return -actor_loss.mean(), ratio

def calc_critic_loss(mini_batch, value, clip_range, batch_return):
    
    clipped_value = mini_batch['values'] + nd.clip(value.reshape(-1) - mini_batch['values'], -clip_range, clip_range)
    critic_loss = nd.concat(((value.reshape(-1)  - batch_return)**2).reshape(1,-1), ((clipped_value - batch_return)**2).reshape(1,-1), dim = 0)
    critic_loss = nd.max(critic_loss, axis = 0)
    return critic_loss.mean()

def calc_entropy_loss(probs):
    
    probs = probs+1e-10
    entropy_loss = -(probs * probs.log()).sum(axis=1)
    return entropy_loss.mean()

def generate_gif_coder_pre(frame_number, frames_for_gif, path, runner):

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx.asnumpy(), (320,420), interpolation=cv2.INTER_LINEAR)

    imageio.mimsave(f'{path}{"ATARI_pre_update_{0}_runner{1}.gif".format(frame_number, runner)}',
                    frames_for_gif, format = 'GIF-PIL', duration=1 / 30)

def generate_gif_coder_enc(frame_number, frames_for_gif, path, runner):

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx.asnumpy(), (320,420), interpolation=cv2.INTER_LINEAR)

    imageio.mimsave(f'{path}{"ATARI_enc_update_{0}_runner{1}.gif".format(frame_number, runner)}',
                    frames_for_gif, format = 'GIF-PIL', duration=1 / 30)

def generate_gif_coder_phos(frame_number, frames_for_gif, path, runner):

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx.asnumpy(), (320,420), interpolation=cv2.INTER_LINEAR)

    imageio.mimsave(f'{path}{"ATARI_phos_update_{0}_runner{1}.gif".format(frame_number, runner)}',
                    frames_for_gif, format = 'GIF-PIL', duration=1 / 30)

def generate_gif_coder_dec(frame_number, frames_for_gif, path, runner):

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx.asnumpy(), (320,420), interpolation=cv2.INTER_LINEAR)

    imageio.mimsave(f'{path}{"ATARI_dec_update_{0}_runner{1}.gif".format(frame_number, runner)}',
                    frames_for_gif, format = 'GIF-PIL', duration=1 / 30)



def generate_gif(frame_number, frames_for_gif, reward, path, step, runner):

    for idx, frame_idx in enumerate(frames_for_gif):

        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_update_{0}_reward_{1}_step_{2}_runner_{3}.gif".format(frame_number, reward, step, runner)}',
                    frames_for_gif, duration=1 / 30)
    

def generate_gif_pre(frame_number, frames_for_gif, reward, path, step, runner):

    for idx, frame_idx in enumerate(frames_for_gif):

        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),preserve_range=True, order=0)

    imageio.mimsave(f'{path}{"ATARI_pre_update_{0}_reward_{1}_step_{2}_runner_{3}.gif".format(frame_number, reward, step, runner)}',
                    frames_for_gif, duration=1 / 30)

def generate_gif_phos(frame_number, frames_for_gif, reward, path, step, runner):

    for idx, frame_idx in enumerate(frames_for_gif):

        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), preserve_range=True, order=0)


    imageio.mimsave(f'{path}{"ATARI_phos_update_{0}_reward_{1}_step_{2}_runner_{3}.gif".format(frame_number, reward, step, runner)}',frames_for_gif, duration=1 / 30)

def generate_gif_enc(frame_number, frames_for_gif, reward, path, step, runner):

    for idx, frame_idx in enumerate(frames_for_gif):

        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), preserve_range=True, order=0) 

    imageio.mimsave(f'{path}{"ATARI_enc_update_{0}_reward_{1}_step_{2}_runner_{3}.gif".format(frame_number, reward, step, runner)}',
                    frames_for_gif, duration=1 / 30)

def generate_gif_dec(frame_number, frames_for_gif, reward, path, step, runner):

    for idx, frame_idx in enumerate(frames_for_gif):

        frames_for_gif[idx] = cv2.normalize(resize(frame_idx, (420, 320, 3),preserve_range=True, order=0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    imageio.mimsave(f'{path}{"ATARI_dec_update_{0}_reward_{1}_step_{2}_runner_{3}.gif".format(frame_number, reward, step, runner)}',frames_for_gif, duration=1 / 30)

    
def perlin_noise_map(seed=0,shape=(h_dec,w_dec),scale=100,octaves=6,persistence=.5,lacunarity=2.):

    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out

