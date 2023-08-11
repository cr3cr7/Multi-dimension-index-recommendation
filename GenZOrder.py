import gym
import os
from gym import spaces
import numpy as np
import subprocess
import random
import shutil
import pickle
import time
import pandas as pd
import sys

import common
import datasets
from util.Block import QueryGeneration
from common import Discretize
import math
from wandb.integration.sb3 import WandbCallback


class GenZOrder(gym.Env):
    
    def __init__(self, dataset='dmv-tiny', queryNum=20, blockSize=20):
        self.table = self.load_data(dataset)
        self._build_col_bits(self.table)
        self.block_size = blockSize
        self.block_nums = math.ceil(self.table.data.shape[0] / self.block_size)
        self.testQuery, self.testScanConds = self.load_query(queryNum, self.table, self.colNames)

        self.action_space = spaces.Discrete(len(self.cols))
        self.observation_space = spaces.MultiDiscrete(np.array([2] * sum(self.col_bits.values())))
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.action_list = []
        # reward_file = os.path.join(self.PATH, 'reward.pkl')
        # if os.path.exists(reward_file):
        #     os.remove(reward_file)
        
    def load_data(self, dataset) ->common.CsvTable:
        if dataset == 'tpch':
            self.table = datasets.LoadTPCH()
        elif dataset == 'dmv-tiny':
            self.table = datasets.LoadDmv('dmv-tiny.csv')
        elif dataset == 'lineitem':
            self.table = datasets.LoadLineitem('lineitem.csv')
        else:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}')
        print(self.table.data.info())
        return self.table
        
    def load_query(self, num, table, colNames):
        # FIMXE: 指定查询的列
        return QueryGeneration(num, self.table.data, colNames)
    
    def _build_col_bits(self, table: common.CsvTable):
        def bits_for_distinct_values(n: int) -> int:
            return math.ceil(math.log2(n))
        
        self.colNames = []
        self.cols = []
        self.col_bits = {}
        self.state = {}
        self.tuples_np = []
        self.tuples_np_orignal = []
        idx = 0
        for col in table.Columns():
            if col.distribution_size > 1:
                self.cols.append(col)
                self.colNames.append(col.name)
                bit_number = bits_for_distinct_values(col.distribution_size)
                self.col_bits[idx] = bit_number
                self.state[idx] = np.zeros(self.col_bits[idx], dtype=np.int32)
                self.tuples_np_orignal.append(Discretize(col))
                self.tuples_np.append([format(value - 1, f'0{bit_number}b') for value in Discretize(col)])   
                idx += 1    
        self.tuples_np = np.stack(self.tuples_np, axis=1)
        self.tuples_np_orignal = np.stack(self.tuples_np_orignal, axis=1)
        # concat mutiple columns
        # 01 00 10 -> 010010
        self.rows = []
        for i in range(len(self.tuples_np)):
            # self.rows.append(int(''.join(self.tuples_np[i]), 2))
            self.rows.append(''.join(self.tuples_np[i]))
        # self.rows = np.asarray(self.rows).astype(np.uint8)
        # print('rows:', self.rows)

        
    def observation(self):
        return np.concatenate([np.atleast_1d(self.state[i]) for i in range(len(self.state))])
        
    def reset(self):
        self.state = {i: np.zeros(self.col_bits[i], dtype=np.int32) for i in range(len(self.col_bits))}
        self.action_list = []
        return self.observation()
        
    def step(self, action):
        try:
            assert self.action_is_valid(action)
        except:
            action = np.argmax(self.valid_action_mask())
            print('new action', action)
        idx = np.argmax(self.state[action] != 1)
        self.state[action][idx] += 1
        self.action_list.append(action)
        
        if self.is_done():
            assert len(self.action_list) == sum(self.col_bits.values())
            return self.observation(), self.get_reward(), True, {}
        else:
            return self.observation(), 0, False, {}    
    
    def action_is_valid(self, action):
        if np.all(self.state[action] == 1):
            return False
        return True
        
    def is_done(self):
        if np.all(self.observation() == 1):
            return True
        return False
    
    def valid_action_mask(self):
        return np.array([self.action_is_valid(i) for i in range(self.action_space.n)])

    def get_reward(self):
        reward_file = os.path.join(self.PATH, 'reward.pkl')
        if os.path.exists(reward_file):
            with open(reward_file, 'rb') as f:
                reward = pickle.load(f)  
        else:
            reward = {}  
        key = ','.join(str(self.action_list))
        if key in reward:
            return reward[key]
        else:
            cur_reward = self.excute()
            reward[key] = cur_reward 
            with open(reward_file, 'wb') as f:
                pickle.dump(reward, f)
            return cur_reward
                
        
    def excute(self):
        # Generate Z order
        # action_list = [1, 1, 0, 0, 2, 2]
        action_list = self._convert_action_list(self.action_list)
        # print('new action list', action_list, len(action_list))
        z_value = interleave_string_columns(self.rows, action_list)
        z_value = [int(x) for x in z_value]
        # z_value = interleave_columns(self.rows, action_list)
        self.table.data['zvalue'] = z_value
        self.table.data.sort_values(by='zvalue', inplace=True)
        self.table.data['id'] = np.arange(self.table.data.shape[0]) // self.block_size
        scan_file = 0
        for id in range(self.block_nums):
            for i in range(len(self.testScanConds)):
                block = common.block(self.table, self.block_size, self.testScanConds[i][0], id)
                # print(id, block.get_data())
                if block.is_scan(self.testScanConds[i][0], self.testScanConds[i][1]):
                    scan_file += 1
            # if block.is_scan(self.testScanConds[0][0], self.testScanConds[0][1]):
            #     scan_file += 1
        # print(z_value)
        # scan_file = 1
        # print(scan_file)
        return -scan_file
    
    def _convert_action_list(self, action_list):
        """convert action into index position of bit in original cols"""
        start_positions = [0] + [self.col_bits[i] for i in range(len(self.col_bits))]
        start_positions = np.cumsum(start_positions)[:-1]
        new_action_list = []
        for action in action_list:
            new_action_list.append(start_positions[action])
            start_positions[action] += 1
        assert len(new_action_list) == len(set(new_action_list))
        return new_action_list



class SelColEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):  # params
        # table涉及的列传入进来，0表示workload不涉及该列，1表示workload涉及该列
        # 比如（1,1,1）workload涉及表中的三列
        # self.aaa = params['']
        ColShow = [1] 
        workload = [13, 25, 17, 20, 15, 10]
        self.rand_num = 0
        self.best_reward = -100
        self.best_actions = []
        self.count = 0
        self.init_reward_selected_cols()
        self.done_col_reward = {}
        self.ColShow = np.array(ColShow)
        self.workload = np.array(workload)
        self.workload_length = len(self.workload)
        # 用来最后计算reward的
        self.length = len(self.ColShow)
        self.SelCol = np.array([0] * self.length)
        self.idx = 0
        self.next_state = np.zeros(self.length)
        # action应该是选择的列，
        self.action_space = spaces.Discrete(2)
        # self.action_space = np.array()
        # self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.length,))
        self.observation_space = spaces.Dict(
            {
                "next_col": spaces.Box(low = 0, high = 1, shape = (self.length,), dtype = int),
                # "workload": spaces.Box(low = 0, high = 100, shape=(self.workload_length,), dtype = int),
            }
        )
        self.all_dataframe = pd.read_csv("/home/ning/zorder/zorder/lineitem_orderAllCols.csv", sep='|')
        #self.sample_dataframe = pd.read_csv("/home/ning/zorder/zorder/lineitem_orderAllCols_sample.csv", sep='|')

    def reset(self, seed=None, options=None):
        self.count += 1
        # super().reset(seed=seed)
        self.idx = 0
        self.SelCol = np.array([0]*self.length)
        self.next_state = np.zeros(self.length)
        self.next_state[self.idx] = 1
        observation = self._get_obs()
        # observation = self.next_state
        return observation

    def step(self, action):
        if self.idx + 1 == len(self.ColShow):
            done = True
            if (self.SelCol.any() == 0):
                reward = -100
                self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
            else:
                self.done_col_reward = self.Get_done_reward()
                if self.count > 500000:
                    if self.done_col_reward:
                        self.best_actions,self.best_reward = self.Get_best_reward_action()
                cases = len(self.done_col_reward)
                if cases >= 1:
                    self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
                    if str(self.SelCol) in self.done_col_reward.keys():
                        reward = self.done_col_reward[str(self.SelCol)]
                    else:
                        self.execu_predicted_files()
                        reward = self.Get_reward()
                        if str(self.SelCol) in self.done_col_reward.keys():
                            pass
                        else:
                            self.done_col_reward[str(self.SelCol)] = reward
                else:
                    while str(self.SelCol) in self.done_col_reward.keys():
                        random1 = np.random.randint(0,100)
                        random2 = np.random.randint(0,100)
                        self.SelCol[random1 % self.length] = random2 % 2
                    self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
                    self.execu_predicted_files()
                    reward = self.Get_reward()
                    if str(self.SelCol) in self.done_col_reward.keys():
                            pass
                    else:
                        self.done_col_reward[str(self.SelCol)] = reward
                self.save_done_reward()      
                reward = str(reward)
                reward = reward.strip('\n')
                reward = float(reward)
                if self.count > 500000:
                    if reward > float(self.best_reward):
                        self.best_reward = reward
                        self.best_actions = self.SelCol.copy()
            self.save_rewards('/home/ning/zorder/New_agg_result/reward.txt',reward)
        else:
            if self.count > 500000:
                if self.best_reward > -100:
                    self.rand_num = random.randint(0,10)
                    if self.rand_num > 3:
                        action = self.best_actions[self.idx]
            done = False
            if action == 0:
                self.SelCol[self.idx] = 0
            else:
                self.SelCol[self.idx] = 1
            self.idx += 1
            self.next_state = self.SelCol.copy()
            self.next_state[self.idx] = 1
            reward = -110
        reward = float(reward)
        if self.count > 500000:
            if reward == self.best_reward:
                # print(reward)
                # print(self.best_actions)
                reward = -reward
        return self._get_obs(), reward, done ,{}
        # return self.next_state, reward, done ,{} 

    def _get_obs(self):
        return {"next_col":self.next_state}
        # return {"next_col":self.next_state,"workload":self.workload}
        # return self.next_state

    def save_col(self,filename):
        # self.SelCol = [0,1,1,0]
        col_string = np.array2string(self.SelCol,separator= ',')
        with open (filename,'a') as f:
            f.write(col_string)
        with open (filename, 'a') as f:
            f.write('\n')
    
    # def get_reward(self):
    #     Sel_Col = str(self.SelCol)
    #     return self.rewards[Sel_Col]
    
    def save_rewards(self,filename,reward):
        reward = str(reward)
        reward = reward.strip('\n')
        with open(filename,'a') as f:
            f.write(reward)
            f.write(" ")
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            f.write('\n')


    # def execu_spark(self):
    #     # cmd = "cd / && spark-submit /home/ning/my_spark/share/CoWorkAlg/execu_query.py"
    #     subprocess.run(['docker', 'exec','-it', 'my_spark-spark-1', '/opt/bitnami/python/bin/python','/opt/share/CoWorkAlg/execu_query.py'])
    def Get_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_reward.txt','r') as f:
            lines = f.readlines()
        reward = lines[-1]
        return reward
    def execu_predicted_files(self):
        # os.system('/bin/bash  /home/ning/zorderlearn/py38/bin/activate') 
        os.chdir('/home/ning/zorderlearn/ValidateScanFileNumbers/')
        # os.system("python /home/ning/zorderlearn/ValidateScanFileNumbers/dim_reduction.py --dataset=tpch --glob='tpch_sample-Dim4-2.3*' --num-queries=2000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --input-encoding=embed --output-encoding=embed")
        #os.system("python /home/ning/zorderlearn/ValidateScanFileNumbers/dim_reduction.py --dataset=dmv --glob='Dmv_sample-Dim4-3.2*' --num-queries=2000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --input-encoding=embed --output-encoding=embed")

        # 将DataFrame对象序列化为字节串
        df1_bytes = pickle.dumps(self.all_dataframe)
        #df2_bytes = pickle.dumps(self.sample_dataframe)

        # 计算两个DataFrame对象的大小
        size1 = np.uint32(len(df1_bytes))
        #size2 = np.uint32(len(df2_bytes))

        proc = subprocess.Popen(['python', '/home/ning/zorderlearn/ValidateScanFileNumbers/dim_reduction.py', '--dataset=tpch','--glob=tpch_sample-Dim4-2.3*', '--num-queries=2000', '--residual', '--layers=5', '--fc-hiddens=256', '--direct-io', '--column-masking', '--input-encoding=embed', '--output-encoding=embed'], stdin=subprocess.PIPE)
        proc.stdin.write(size1.tobytes())
        proc.stdin.write(df1_bytes)
        #proc.stdin.write(size2.tobytes())
        #proc.stdin.write(df2_bytes)
        proc.stdin.close()
        proc.wait()

    def Get_Single_min_ratio(self):
        with open('/home/ning/zorder/Actions_Rewards/single_min_select_ratio.txt','r') as f:
            lines = f.readlines()
        reward = lines[-1]
        return reward
    def mkfile(self):
        open(self.parent_path + '/' + 'rewards.txt','w')
        open(self.parent_path + '/' + 'Select_cols.txt','w')
        open(self.parent_path + '/' + 'workload_erows_ratio.txt','w')
        self.dir_path = self.parent_path + str(self.define_col_num)
        folder = os.path.exists(self.dir_path)
        if folder:
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)
        # open(self.dir_path + '/' + 'rewards.txt','w')
        # open(self.dir_path + '/' + 'Select_cols.txt','w')
        # open(self.dir_path + '/' + 'workload_erows_ratio.txt','w')
        # return self.dir_path
    def save_done_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_epsiode.pkl','wb') as f:
            pickle.dump(self.done_col_reward,f)
    def Get_done_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_epsiode.pkl','rb') as f:
            self.done_col_reward = pickle.load(f)
            return self.done_col_reward
    def init_reward_selected_cols(self):
        with open('/home/ning/zorder/New_agg_result/reward.txt','w') as f:
            f.write('')
        with open('/home/ning/zorder/New_agg_result/select_cols.txt','w') as f:
            f.write('')
    def Get_best_reward_action(self):
        done_col_reward = list(self.done_col_reward.items())
        best_action = done_col_reward[0][0]
        best_reward = float(str(done_col_reward[0][1]).strip('\n'))
        for i in range(len(done_col_reward)):
            temp_reward = float(str(done_col_reward[i][1]).strip('\n'))
            if temp_reward > best_reward:
                best_reward = temp_reward
                best_action = done_col_reward[i][0]
        best_action = ','.join(str(best_action).split())
        best_action = eval(best_action)
        best_action = np.array(best_action)
        print(type(best_action))
        print(best_reward)
        return best_action,best_reward



def interleave_string_columns(data, action_list):
    result = np.empty(len(data), dtype='U48')
    for idx, one_row in enumerate(data):
        result[idx] = int(''.join([one_row[i] for i in action_list]), 2)
    return result
        

def interleave_columns(data, action_list):
    # Define a lookup table that maps column indices to data arrays
    # lookup = {i: data[:, i] for i in range(data.shape[1])}
    # lookup = np.packbits(data, axis=1)
    all_len = len(action_list)
    # Interleave the data based on the action list
    interleaved = np.zeros(data.shape[0], dtype=np.uint8)
    for i, col_idx in enumerate(action_list):
        # col_data = lookup
        bit_pos = col_idx
        bits = (data >> (all_len - bit_pos - 1)) & 1
        interleaved |= bits << (all_len - i - 1)
    
    return interleaved


def concatenate_columns(data, column_indices):
    # Initialize an empty array to hold the output values
    output_values = np.zeros(data.shape[0], dtype=np.uint32)
    
    # Loop over each row of the input table
    for i, row in enumerate(data):
        # Select the columns that you want to concatenate
        selected_columns = row[column_indices]
        
        # Convert each column to a numpy array of binary digits
        binary_arrays = [np.unpackbits(col.astype(np.uint8)) for col in selected_columns]
        
        # Combine the binary arrays into a single binary array
        combined_array = np.concatenate(binary_arrays)
        
        # Convert the combined binary array back to a single integer
        output_value = int(''.join(map(str, combined_array)), 2)
        
        # Store the output value in the output array
        output_values[i] = output_value
    
    return output_values

def concatenate_columns(data, column_indices):
    # Select the columns that you want to concatenate
    selected_columns = data[:, column_indices]
    
    # Convert each column to a numpy array of binary digits
    binary_arrays = np.unpackbits(selected_columns.astype(np.uint8), axis=1)
    # print(binary_arrays)
    # Combine the binary arrays into a single binary array
    combined_array = np.concatenate(binary_arrays, axis=0)
    
    # Convert the combined binary array back to a single integer
    output_values = np.packbits(combined_array)
    
    return output_values


def register_env():
    from gym.envs.registration import register

    environments = [['GenZOrder', 'v0']]

    for environment in environments:
        register(
            id='{}-{}'.format(environment[0], environment[1]),
            entry_point='GenZOrder:{}'.format(environment[0]),
            nondeterministic=True
    )


if __name__ == "__main__":
    # TODO: implementation of maskable ppo in ray RLlib
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    
    from sb3_contrib.ppo_mask import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.monitor import Monitor
    import wandb
    from stable_baselines3.common.env_util import make_vec_env
    
    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.valid_action_mask()
    
    def make_env(): 
        env = GenZOrder('dmv-tiny', 20, 20)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    
    register_env()
    # env = gym.make("GenZOrder-v0")
    env = GenZOrder('dmv-tiny', 20, 20)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    # async_env = gym.vector.AsyncVectorEnv([ lambda: make_env(),
    #                                         lambda: make_env(),
    #                                         lambda: make_env(),
    #                                         lambda: make_env()
    #                                     ])
    # async_env = make_env()
    # async_env = make_vec_env('GenZOrder-v0', n_envs=1)
    # async_env = ActionMasker(async_env, mask_fn)
    # print(env.reset())
    
    config = {
    "policy_type": "MaskableActorCriticPolicy",
    "total_timesteps": 2048 * 10000,
    "env_name": "Zorder",
    # "WandBCallback": None,
    # "tensorboard_log": None
    }
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    
    # callback=WandbCallback(gradient_save_freq=100, verbose=2)
    # tensorboard_log=f"lightning_logs/{run.id}"
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1,tensorboard_log=f"lightning_logs/{run.id}")
    model.learn(total_timesteps=config['total_timesteps'], callback=WandbCallback(gradient_save_freq=100, verbose=2))
    run.finish()
    
    # env.reset()
    # for i in range(100):
    #     action = env.action_space.sample()
    #     # print(action)
    #     observation, reward, done, _ = env.step(action)
    #     if done:
    #         break
        
    
    
    """ action_list = [2,3,0,1,4,5]
    # action_list = [0,1,2,3,4,5]
    data = np.array([['10', '10', '11'],
                    ['11', '10', '01']])

    data_1 = np.apply_along_axis(lambda d: int(d[0] + d[1] + d[2], 2), 1, data).astype(np.uint8)
    # print(bin(data_1[0]))
    # print(bin(data_1[1]))
    
    column_indices = [0, 1, 2]
    
    output_values = interleave_columns(data_1, action_list)
    print(bin(output_values[0]))
    print(bin(output_values[1])) """
