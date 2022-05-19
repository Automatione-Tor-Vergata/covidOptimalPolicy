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

[map, targets] = two_people_map();
covid_two_env = COVIDGridworld(2, map, targets, {'r', 'g'}, 0.2);
covid_two_env.num_cells = size(map, 1) * size(map, 2);

validateEnvironment(covid_two_env);
covid_two_env.reset();

maxNumCompThreads(6); % Limit CPU cores usage



%% Create the training algorithm.
sarsa_agent = makeCriticAgent(covid_two_env);

load sarsaTrain.mat
load critic_params_trained.mat

critic = getCritic(sarsa_agent);
critic = setLearnableParameters(critic,criticParams);
setCritic(sarsa_agent,critic);


trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',25,...
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

trainStats = train(sarsa_agent,covid_two_env,trainOpts);

% Extract Weight of the network
critic = getCritic(sarsa_agent);
criticParams = getLearnableParameters(critic);

save("sarsaTrain.mat",'trainStats','covid_two_env','trainOpts');

save("critic_params_trained",'criticParams');



