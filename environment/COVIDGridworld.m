% Multiple people gridworld implementation for COVID avoidance policy.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 13, 2021

classdef COVIDGridworld < rl.env.MATLABEnvironment
    %COVIDGRIDWORLD: Multiple people RL gridworld.

    %% Gridworld Properties
    properties
        % Environment constants and physical characteristics.
        % People in the map.
        n_people
        
        % Matrix encoding the map state in real time.
        map_mat
        
        % Target positions indices (depends on n_people).
        targets
        
        % Stall-leading actions counter.
        stall_acts_cnt = 0
        
        % Stall leading actions counter max value.
        max_stall_acts = 50
        
        % "Defeat" state reward (depends on map size).
        defeat_rew = 0
        
        % Single step reward.
        single_step_rew = -1
        
        % "Victory" state base reward.
        victory_rew = 0

        % COVID-19 infected people flags.
        infected_people

        % Starting number of COVID-19 infected.
        infected_init

        % COVID-19 contagion probability.
        contagion_prob

        % COVID-19 infected delta reward multiplier.
        infected_delta_gain = -100

        % COVID-19 contagion risk zone radius.
        contagion_zone_radius = 1
        
        % People positions (depends on n_people).
        State

        % Internal flag to indicate episode termination.
        IsDone = false

        % Plot figure handle.
        Figure

        % Plot figure axes.
        Ax

        % Plot grid lines.
        GridLines

        % Handles for people plot patches.
        PeoplePatches

        % Array of strings that specify colors to plot different people.
        Colors
    end

    %% Necessary Methods
    methods
        function this = COVIDGridworld(people, map, target_indices, colors, covid_prob)
        % COVIDGridworld    Creates an instance of the environment.
            
            % Generate a cell array that holds all possible actions.
            % Works as a car odometer.
            actions_cell = cell([5 ^ people, 1]);
            prev_cell = ones(people, 1);
            actions_cell{1} = prev_cell;
            for i = 2:(5 ^ people)
                prev_cell = actions_cell{i - 1};
                for j = 1:people
                    prev_cell(j) = prev_cell(j) + 1;
                    if prev_cell(j) == 6
                        prev_cell(j) = 1;
                        continue
                    else
                        break
                    end
                end
                actions_cell{i} = prev_cell;
            end
            
            % Initialize Observation settings.
            ObservationInfo = rlNumericSpec([people 1]);
            ObservationInfo.Name = 'COVIDGridworld Observation';
            ObservationInfo.Description = 'People positions';
            ObservationInfo.LowerLimit = 0;
            ObservationInfo.UpperLimit = size(map, 1) * size(map, 2);
            
            % Initialize Action settings.
            ActionInfo = rlFiniteSetSpec(actions_cell);
            ActionInfo.Name = 'COVIDGridworld Action';
            ActionInfo.Description = 'Set of people STOP+NSWE movements';
            
            % The following line implements built-in functions of RL env.
            % NOTE: This MUST be called before setting anything else!
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Initialize other properties.
            this.n_people = people;
            this.map_mat = map;
            this.targets = target_indices;
            this.State = zeros(people, 1);
            this.defeat_rew = -1000 * size(map, 1) * size(map, 2);
            this.Colors = colors;
            this.contagion_prob = covid_prob;
        end

        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
        % STEP  Simulates the environment with the given action.
            LoggedSignals = [];
            all_still = true;
            defeated = false;
            this.IsDone = false;
            
            % First, check for contagion.
            for i = 1:this.n_people
                curr_pos = this.State(i);
                [curr_row, curr_col] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], curr_pos);
                for r = (curr_row - this.contagion_zone_radius):(curr_row + this.contagion_zone_radius)
                    if (r < 1) || (r > size(this.map_mat, 1))
                        continue
                    end
                    for c = (curr_col - this.contagion_zone_radius):(curr_col + this.contagion_zone_radius)
                        if (c < 1) || (c > size(this.map_mat, 2))
                            continue
                        end
                        if (this.map_mat(r, c) > 2) && (this.infected_people(this.map_mat(r, c) - 2) == 1)
                            % An infected is nearby: contagion is now
                            % possible.
                            if binornd(1, this.contagion_prob) == 1
                                this.infected_people(i) = 1;
                            end
                        end
                    end
                end
            end
            
            % Process each person's movements.
            for i = 1:this.n_people
                curr_moved = false;
                curr_pos = this.State(i);
                [curr_row, curr_col] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], curr_pos);
                
                % Parse the action.
                switch Action(i)
                    case 1
                        % STOP
                        new_subs = [curr_row, curr_col];
                        new_pos = curr_pos;
                    case 2
                        % NORTH
                        new_subs = [curr_row - 1, curr_col];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 3
                        % SOUTH
                        new_subs = [curr_row + 1, curr_col];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 4
                        % WEST
                        new_subs = [curr_row, curr_col - 1];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 5
                        % EAST
                        new_subs = [curr_row, curr_col + 1];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    otherwise
                        error('Invalid action %d for person %d.', Action(i), i);
                end
                
                % Check if the new position is feasible.
                if curr_moved == true
                    to_update = false;
                    switch this.map_mat(new_subs(1), new_subs(2))
                        case 1
                            % Free cell: action is legit.
                            to_update = true;
                        case 2
                            % Obstacle detected: did someone got here in
                            % the meantime?
                            if this.map_mat(curr_row, curr_col) ~= 2 + i
                                defeated = true;
                                break
                            end
                            % If control got here, there's nothing to do.
                        otherwise
                            % Occupied cell: who got here first?
                            other_guy = this.map_mat(new_subs(1), new_subs(2)) - 2;
                            if other_guy < i
                                % He got here first: illegal collision.
                                defeated = true;
                                break
                            else
                                % What's he going to do?
                                if Action(other_guy) == 1
                                    % He's here to stay: illegal collision.
                                    defeated = true;
                                    break
                                else
                                    % Watch out for illegal crossings.
                                    switch Action(i)
                                        case 2
                                            % NORTH: Is he going SOUTH?
                                            if Action(other_guy) == 3
                                                defeated = true;
                                                break
                                            end
                                        case 3
                                            % SOUTH: Is he going NORTH?
                                            if Action(other_guy) == 2
                                                defeated = true;
                                                break
                                            end
                                        case 4
                                            % WEST: Is he going EAST?
                                            if Action(other_guy) == 5
                                                defeated = true;
                                                break
                                            end
                                        case 5
                                            % EAST: Is he going WEST?
                                            if Action(other_guy) == 4
                                                defeated = true;
                                                break
                                            end
                                    end
                                    % If control got here means the
                                    % action is legit.
                                    to_update = true;
                                end
                            end
                    end
                    
                    % If necessary, update map and internal state.
                    if to_update == true
                        all_still = false;
                        if this.map_mat(curr_row, curr_col) == 2 + i
                            % Must leave the cell only if no one
                            % else got in in the meantime.
                            this.map_mat(curr_row, curr_col) = 1;
                        end
                        this.map_mat(new_subs(1), new_subs(2)) = 2 + i;
                        this.State(i) = new_pos;
                    end
                end
            end
            
            % Has no one moved?
            if all_still == true
                this.stall_acts_cnt = this.stall_acts_cnt + 1;
                if this.stall_acts_cnt == this.max_stall_acts
                    % Stalled for too long.
                    defeated = true;
                end
            else
                this.stall_acts_cnt = 0;
            end
            
            % "Defeat" state: set return values and get out.
            if defeated == true
                this.IsDone = true;
                IsDone = true;
                Observation = zeros(this.n_people, 1);
                Reward = this.defeat_rew;
                notifyEnvUpdated(this);
                return
            end
            
            % Check for "Victory".
            won = true;
            for i = 1:this.n_people
                if this.State(i) ~= this.targets(i)
                    won = false;
                end
            end
            if won == true
                this.IsDone = true;
                IsDone = true;
                Observation = this.State;
                Reward = this.victory_rew + this.infected_delta_gain * (sum(this.infected_people) - this.infected_init);
                notifyEnvUpdated(this);
                return
            end
            
            % Just a normal execution step.
            IsDone = false;
            Observation = this.State;
            Reward = this.single_step_rew;
            notifyEnvUpdated(this);
        end

        function InitialObservation = reset(this)
        % RESET Resets environment and observation to initial state.
            
            % Reset counters and other properties.
            this.stall_acts_cnt = 0;
            this.IsDone = false;
            
            % Clear the map from people (not the first time!).
            if this.State(1) ~= 0
                for i = 1:this.n_people
                    [person_row, person_col] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], this.State(i));
                    this.map_mat(person_row, person_col) = 1;
                end
            end
            
            % Generate and set a new initial state.
            InitialObservation = zeros(this.n_people, 1);
            for i = 1:this.n_people
                while true
                    new_pos = randi(size(this.map_mat, 1) * size(this.map_mat, 2));
                    % Check if this is not a target.
                    if ismember(new_pos, this.targets) == true
                        % A new random extraction is necessary.
                        continue
                    end
                    % Check if the cell is free.
                    [new_row, new_col] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], new_pos);
                    if this.map_mat(new_row, new_col) ~= 1
                        % A new random extraction is necessary.
                        continue
                    else
                        this.map_mat(new_row, new_col) = 2 + i;
                        this.State(i) = new_pos;
                        InitialObservation(i) = new_pos;
                        break
                    end
                end
            end
            
            % Set initially infected people.
            this.infected_people = zeros(this.n_people, 1);
            this.infected_init = randi(ceil(0.5 * this.n_people));
            to_infect = randperm(this.n_people, this.infected_init);
            for i = to_infect
                this.infected_people(i) = 1;
            end
            
            % Signal that the environment has been updated.
            notifyEnvUpdated(this);
        end
    end

    %% Auxiliary Methods
    methods
        function plot(this)
            % PLOT  Creates a visualization of the environment.
            if isempty(this.Figure) || ~ishandle(this.Figure)
                % Create the figure.
                this.Figure = figure('MenuBar','none','Toolbar','none', 'Visible', 'on', 'Name', 'COVIDGridworld');
                this.Ax = axes('Parent', this.Figure);
                this.Ax.Toolbar.Visible = 'off';
                this.Ax.Visible = 'off';
                axis(this.Ax, 'equal')
                hold(this.Ax, 'on');
            end
            
            % Initialize people patches handles array.
            this.PeoplePatches = [];
            
            % Draw grid.
            delete(this.GridLines);
            m = size(this.map_mat, 1);
            n = size(this.map_mat, 2);
            xLineData = [];
            yLineData = [];
            y0 = 0.5;
            x0 = 0.5;
            for r = 0:m
                yLineData = [yLineData; y0 + r; y0 + r; nan];
                xLineData = [xLineData; x0; n + 0.5; nan];
            end
            for c = 0:n
                xLineData = [xLineData; x0 + c; x0 + c; nan];
                yLineData = [yLineData; y0; m + 0.5; nan];
            end
            this.GridLines = plot(this.Ax, xLineData, -yLineData);
            
            % Draw obstacles.
            for r = 1:size(this.map_mat, 1)
                for c = 1:size(this.map_mat, 2)
                    if this.map_mat(r, c) == 2
                        X = [c - 0.5, c + 0.5, c + 0.5, c - 0.5, c - 0.5];
                        Y = -[r - 0.5, r - 0.5, r + 0.5, r + 0.5, r - 0.5];
                        patch(this.Ax, X, Y, 'k');
                    end
                end
            end
            
            % Draw targets.
            for i = 1:length(this.targets)
                [r, c] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], this.targets(i));
                X = [c - 0.5, c + 0.5, c + 0.5, c - 0.5, c - 0.5];
                Y = -[r - 0.5, r - 0.5, r + 0.5, r + 0.5, r - 0.5];
                patch(this.Ax, X, Y, this.Colors(i));
            end
            
            % Update the visualization.
            envUpdatedCallback(this);
        end
    end

    methods (Access = protected)
        function envUpdatedCallback(this)
        % ENVUPDATEDCALLBACK    Updates the environment visualization.
            if ~isempty(this.Figure) && isvalid(this.Figure)
                Data = 0.3 * exp(1j * (0:.2:2*pi+.2));
                for i = 1:this.n_people
                    if length(this.PeoplePatches) == this.n_people
                        delete(this.PeoplePatches(i));
                    end
                    [curr_row, curr_col] = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], this.State(i));
                    XData = curr_col + real(Data);
                    YData = -curr_row + imag(Data);
                    this.PeoplePatches(i) = patch(this.Ax, XData, YData, this.Colors(i));
                end
                
                % Refresh rendering in the figure window.
                drawnow
            end
        end
    end
end