% Script to generate a 3-people environment and train a SARSA agent in it.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% April 19, 2022



% Create, validate and reset the new environment.

clearvars
close all
clc

rng(42);

[map, targets] = three_people_map();
covid_three_env = COVIDGridworld(3, map, targets, {'r', 'g', 'b'}, 0.2);
covid_three_env.num_cells = size(map, 1) * size(map, 2);

validateEnvironment(covid_three_env);
covid_three_env.reset();

maxNumCompThreads(6); % Limit CPU cores usage

%% Create the training algorithm.
sarsa_agent = makeCriticAgent(covid_three_env);

load sarsaTrain.mat

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',80,...
    'MaxStepsPerEpisode',100,...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',0, ...
    'Verbose',true,...
    'Plots',"training-progress");

% trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "gradients"; %for A3C
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 20;
trainOpts.ParallelizationOptions.WorkerRandomSeeds = -1;
trainOpts.StopOnError = 'off';

%% Train the agent in the environment.
% plot(covid_three_env);

trainStats = train(sarsa_agent,covid_three_env,trainOpts);

save("sarsaTrain.mat",'trainStats','covid_three_env','trainOpts');

% Extract Weight of the network
critic = getCritic(sarsa_agent);
criticParams = getLearnableParameters(critic);

% %% RESUME
% sarsa_agent = makeCriticAgent(covid_three_env);
% load sarsaTrain.mat
% trainStats = train(sarsa_agent,covid_three_env,trainOpts);
% save("sarsaTrain.mat",'trainStats','covid_three_env','trainOpts');
% 
% % Extract Weight of the network
% critic = getCritic(sarsa_agent);
% criticParams = getLearnableParameters(critic);
