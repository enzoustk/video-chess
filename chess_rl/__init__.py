"""Pacote do agente de Deep RL para o ambiente Atari Video Chess (ALE/Farama).

Módulos:
    board    -> decodificação do tabuleiro e balanço material a partir da RAM
    encoding -> transformação RAM -> representação de estado (planos + auxiliares)
    env      -> criação do ambiente e wrapper de modelagem de recompensa
    network  -> arquitetura Dueling DQN (CNN do tabuleiro + MLP auxiliar)
    replay   -> Experience Replay (buffer circular)
    agent    -> agente Double/Dueling DQN com alvo, Huber loss e clipping
    config   -> hiperparâmetros
"""
__all__ = ["board", "encoding", "env", "network", "replay", "agent", "config"]
