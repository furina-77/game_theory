#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机辅助移动边缘计算中的任务卸载激励机制
基于博弈论的隐私保护在线任务卸载
"""

import numpy as np
import random
from typing import List, Dict, Tuple

class User:
    """用户类，表示需要卸载任务的移动设备"""
    def __init__(self, user_id: int, location: Tuple[float, float], 
                 task_data_size: float, task_complexity: float, 
                 local_computing_speed: float, privacy_sensitivity: float):
        """
        初始化用户
        
        Args:
            user_id: 用户ID
            location: 用户位置坐标 (x, y)
            task_data_size: 任务数据大小 (MB)
            task_complexity: 任务复杂度 (CPU周期/bit)
            local_computing_speed: 本地计算速度 (CPU周期/秒)
            privacy_sensitivity: 隐私敏感度 (0-1，值越大越敏感)
        """
        self.user_id = user_id
        self.location = location
        self.task_data_size = task_data_size
        self.task_complexity = task_complexity
        self.local_computing_speed = local_computing_speed
        self.privacy_sensitivity = privacy_sensitivity
        
        # 计算本地计算延迟和能耗
        self.local_delay = (task_data_size * 1024 * 8 * task_complexity) / local_computing_speed
        self.local_energy = self._calculate_local_energy()
        
        # 初始化卸载决策和支付
        self.offloading_decision = False
        self.payment = 0.0
    
    def _calculate_local_energy(self) -> float:
        """
        计算本地计算能耗
        
        Returns:
            本地计算能耗 (J)
        """
        # 简化模型：能耗 = 计算量 * 功耗系数
        power_coefficient = 1e-27  # 功耗系数
        total_cycles = self.task_data_size * 1024 * 8 * self.task_complexity
        return power_coefficient * (self.local_computing_speed ** 3) * (total_cycles / self.local_computing_speed)
    
    def calculate_offloading_delay(self, uav: 'UAV') -> float:
        """
        计算卸载到无人机的延迟
        
        Args:
            uav: 目标无人机
        
        Returns:
            卸载延迟 (秒)
        """
        # 计算传输延迟
        distance = np.sqrt((self.location[0] - uav.location[0]) ** 2 + (self.location[1] - uav.location[1]) ** 2)
        transmission_rate = uav.calculate_transmission_rate(distance)
        transmission_delay = self.task_data_size / transmission_rate
        
        # 计算无人机处理延迟
        processing_delay = (self.task_data_size * 1024 * 8 * self.task_complexity) / uav.computing_speed
        
        return transmission_delay + processing_delay
    
    def calculate_offloading_energy(self, uav: 'UAV') -> float:
        """
        计算卸载到无人机的能耗
        
        Args:
            uav: 目标无人机
        
        Returns:
            卸载能耗 (J)
        """
        # 简化模型：传输能耗 = 数据大小 * 传输功率
        transmission_power = 0.1  # 传输功率 (W)
        distance = np.sqrt((self.location[0] - uav.location[0]) ** 2 + (self.location[1] - uav.location[1]) ** 2)
        transmission_rate = uav.calculate_transmission_rate(distance)
        transmission_time = self.task_data_size / transmission_rate
        
        return transmission_power * transmission_time
    
    def calculate_utility(self, uav: 'UAV', payment: float) -> float:
        """
        计算用户效用
        
        Args:
            uav: 目标无人机
            payment: 用户支付给无人机的费用
        
        Returns:
            用户效用值
        """
        # 计算卸载的收益（本地成本 - 卸载成本）
        offloading_delay = self.calculate_offloading_delay(uav)
        offloading_energy = self.calculate_offloading_energy(uav)
        
        # 延迟成本系数和能耗成本系数
        delay_coefficient = 1.0
        energy_coefficient = 100.0
        
        # 隐私保护成本（基于隐私敏感度）
        privacy_cost = self.privacy_sensitivity * 0.5 * payment
        
        # 本地处理成本
        local_cost = delay_coefficient * self.local_delay + energy_coefficient * self.local_energy
        
        # 卸载处理成本
        offloading_cost = delay_coefficient * offloading_delay + energy_coefficient * offloading_energy + payment + privacy_cost
        
        # 效用 = 本地成本 - 卸载成本（如果卸载更优则为正）
        return local_cost - offloading_cost


class UAV:
    """无人机类，表示提供边缘计算服务的无人机"""
    def __init__(self, uav_id: int, location: Tuple[float, float], 
                 computing_speed: float, bandwidth: float, 
                 power_budget: float, pricing_strategy: float):
        """
        初始化无人机
        
        Args:
            uav_id: 无人机ID
            location: 无人机位置坐标 (x, y)
            computing_speed: 计算速度 (CPU周期/秒)
            bandwidth: 可用带宽 (Mbps)
            power_budget: 功率预算 (W)
            pricing_strategy: 定价策略参数
        """
        self.uav_id = uav_id
        self.location = location
        self.computing_speed = computing_speed
        self.bandwidth = bandwidth
        self.power_budget = power_budget
        self.pricing_strategy = pricing_strategy
        
        # 初始化已分配的计算资源和收入
        self.assigned_computing = 0.0
        self.revenue = 0.0
        
        # 服务的用户列表
        self.served_users: List[User] = []
    
    def calculate_transmission_rate(self, distance: float) -> float:
        """
        计算传输速率
        
        Args:
            distance: 传输距离 (m)
        
        Returns:
            传输速率 (Mbps)
        """
        # 简化的无线传输模型：自由空间传播模型
        frequency = 2.4e9  # 载波频率 (Hz)
        speed_of_light = 3e8  # 光速 (m/s)
        wavelength = speed_of_light / frequency
        
        # 发射功率 (W)
        tx_power = 0.1
        # 路径损耗系数
        path_loss_exponent = 2.0
        # 噪声功率 (W)
        noise_power = 1e-10
        
        # 路径损耗计算
        path_loss = (4 * np.pi * distance / wavelength) ** path_loss_exponent
        
        # 接收功率 (W)
        rx_power = tx_power / path_loss
        
        # 信干噪比 (SNR)
        snr = rx_power / noise_power
        
        # 香农公式计算传输速率 (bps)
        rate_bps = self.bandwidth * 1e6 * np.log2(1 + snr)
        
        # 转换为 Mbps
        return rate_bps / 1e6
    
    def calculate_processing_cost(self, user: User) -> float:
        """
        计算处理用户任务的成本
        
        Args:
            user: 用户
        
        Returns:
            处理成本 (J)
        """
        # 简化模型：能耗成本
        power_coefficient = 1e-27
        total_cycles = user.task_data_size * 1024 * 8 * user.task_complexity
        energy_cost = power_coefficient * (self.computing_speed ** 3) * (total_cycles / self.computing_speed)
        
        # 计算资源占用成本
        resource_cost = 0.1 * (total_cycles / self.computing_speed)
        
        return energy_cost + resource_cost
    
    def calculate_profit(self, user: User, payment: float) -> float:
        """
        计算服务用户的利润
        
        Args:
            user: 用户
            payment: 用户支付的费用
        
        Returns:
            利润值
        """
        processing_cost = self.calculate_processing_cost(user)
        return payment - processing_cost
    
    def update_assigned_computing(self):
        """
        更新已分配的计算资源
        """
        self.assigned_computing = sum(
            user.task_data_size * 1024 * 8 * user.task_complexity 
            for user in self.served_users
        )
    
    def can_serve(self, user: User) -> bool:
        """
        检查是否有足够资源服务新用户
        
        Args:
            user: 用户
        
        Returns:
            是否可以服务
        """
        required_cycles = user.task_data_size * 1024 * 8 * user.task_complexity
        return (self.assigned_computing + required_cycles) <= self.computing_speed


class IncentiveMechanism:
    """激励机制类，实现基于博弈论的任务卸载激励机制"""
    def __init__(self, users: List[User], uavs: List[UAV]):
        """
        初始化激励机制
        
        Args:
            users: 用户列表
            uavs: 无人机列表
        """
        self.users = users
        self.uavs = uavs
    
    def match_users_to_uavs(self) -> Dict[UAV, List[User]]:
        """
        用户-无人机匹配算法
        
        Returns:
            匹配结果，字典形式：无人机 -> [用户列表]
        """
        # 初始化匹配结果
        matching = {uav: [] for uav in self.uavs}
        
        # 简化的匹配算法：基于距离的贪婪匹配
        for user in self.users:
            # 计算用户到所有无人机的距离
            uav_distances = [(uav, np.sqrt(
                (user.location[0] - uav.location[0]) ** 2 + 
                (user.location[1] - uav.location[1]) ** 2
            )) for uav in self.uavs]
            
            # 按距离排序
            uav_distances.sort(key=lambda x: x[1])
            
            # 找到第一个有足够资源的无人机
            for uav, _ in uav_distances:
                if uav.can_serve(user):
                    matching[uav].append(user)
                    uav.served_users.append(user)
                    uav.update_assigned_computing()
                    break
        
        return matching
    
    def calculate_payment(self, user: User, uav: UAV) -> float:
        """
        计算用户支付给无人机的费用
        
        Args:
            user: 用户
            uav: 无人机
        
        Returns:
            支付费用
        """
        # 基于成本加利润的定价策略
        processing_cost = uav.calculate_processing_cost(user)
        
        # 考虑用户隐私敏感度调整价格
        privacy_factor = 1 + user.privacy_sensitivity * 0.5
        
        # 计算支付
        payment = processing_cost * (1 + uav.pricing_strategy) * privacy_factor
        
        return max(payment, 0.01)  # 确保支付不低于最小值
    
    def run_mechanism(self) -> Tuple[Dict[UAV, List[User]], float, float]:
        """
        运行激励机制
        
        Returns:
            匹配结果, 总用户效用, 总无人机利润
        """
        # 1. 用户-无人机匹配
        matching = self.match_users_to_uavs()
        
        total_user_utility = 0.0
        total_uav_profit = 0.0
        
        # 2. 计算支付和效用
        for uav, users in matching.items():
            for user in users:
                # 计算支付
                payment = self.calculate_payment(user, uav)
                user.payment = payment
                
                # 计算用户效用
                user_utility = user.calculate_utility(uav, payment)
                total_user_utility += user_utility
                
                # 更新用户卸载决策
                if user_utility > 0:
                    user.offloading_decision = True
                    # 计算无人机利润
                    uav_profit = uav.calculate_profit(user, payment)
                    total_uav_profit += uav_profit
                    uav.revenue += payment
                else:
                    user.offloading_decision = False
                    # 如果效用为负，不卸载，从服务列表中移除
                    uav.served_users.remove(user)
                    uav.update_assigned_computing()
        
        return matching, total_user_utility, total_uav_profit


def generate_users(num_users: int) -> List[User]:
    """
    生成用户列表
    
    Args:
        num_users: 用户数量
    
    Returns:
        用户列表
    """
    users = []
    for i in range(num_users):
        # 随机生成用户位置 (0-1000m)
        location = (random.uniform(0, 1000), random.uniform(0, 1000))
        # 随机生成任务数据大小 (1-10MB)
        task_data_size = random.uniform(1, 10)
        # 随机生成任务复杂度 (100-1000 CPU周期/bit)
        task_complexity = random.uniform(100, 1000)
        # 随机生成本地计算速度 (1e8-1e9 CPU周期/秒)
        local_computing_speed = random.uniform(1e8, 1e9)
        # 随机生成隐私敏感度 (0-1)
        privacy_sensitivity = random.uniform(0, 1)
        
        user = User(i, location, task_data_size, task_complexity, 
                   local_computing_speed, privacy_sensitivity)
        users.append(user)
    
    return users


def generate_uavs(num_uavs: int) -> List[UAV]:
    """
    生成无人机列表
    
    Args:
        num_uavs: 无人机数量
    
    Returns:
        无人机列表
    """
    uavs = []
    for i in range(num_uavs):
        # 随机生成无人机位置 (0-1000m)
        location = (random.uniform(0, 1000), random.uniform(0, 1000))
        # 随机生成计算速度 (1e9-5e9 CPU周期/秒)
        computing_speed = random.uniform(1e9, 5e9)
        # 随机生成带宽 (10-100 Mbps)
        bandwidth = random.uniform(10, 100)
        # 随机生成功率预算 (10-50 W)
        power_budget = random.uniform(10, 50)
        # 随机生成定价策略 (0.1-0.5)
        pricing_strategy = random.uniform(0.1, 0.5)
        
        uav = UAV(i, location, computing_speed, bandwidth, 
                 power_budget, pricing_strategy)
        uavs.append(uav)
    
    return uavs


def main():
    """
    主函数，演示激励机制运行
    """
    # 设置随机种子，确保结果可复现
    random.seed(42)
    
    # 生成用户和无人机
    num_users = 20
    num_uavs = 5
    
    print("=== 生成用户和无人机 ===")
    users = generate_users(num_users)
    uavs = generate_uavs(num_uavs)
    
    print(f"生成了 {num_users} 个用户和 {num_uavs} 个无人机")
    
    # 创建并运行激励机制
    print("\n=== 运行激励机制 ===")
    incentive_mechanism = IncentiveMechanism(users, uavs)
    matching, total_user_utility, total_uav_profit = incentive_mechanism.run_mechanism()
    
    # 统计卸载率
    offloading_users = sum(1 for user in users if user.offloading_decision)
    offloading_rate = offloading_users / num_users * 100
    
    print(f"\n=== 机制运行结果 ===")
    print(f"总用户效用: {total_user_utility:.2f}")
    print(f"总无人机利润: {total_uav_profit:.2f}")
    print(f"卸载率: {offloading_rate:.1f}%")
    print(f"成功卸载的用户数: {offloading_users}/{num_users}")
    
    # 打印每个无人机服务的用户数
    print("\n=== 无人机服务情况 ===")
    for uav in uavs:
        served_count = len([user for user in uav.served_users if user.offloading_decision])
        print(f"无人机 {uav.uav_id} 服务了 {served_count} 个用户，收入: {uav.revenue:.2f}")
    
    # 打印部分用户详情
    print("\n=== 用户详情 (前5个) ===")
    for i, user in enumerate(users[:5]):
        print(f"用户 {user.user_id}:")
        print(f"  位置: {user.location}")
        print(f"  隐私敏感度: {user.privacy_sensitivity:.2f}")
        print(f"  本地延迟: {user.local_delay:.4f}s")
        print(f"  卸载决策: {'是' if user.offloading_decision else '否'}")
        print(f"  支付费用: {user.payment:.2f}")
        print()


if __name__ == "__main__":
    main()
