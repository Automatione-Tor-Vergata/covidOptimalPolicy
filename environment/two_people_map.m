% 2-people map generation script.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% April 19, 2022


function [grid, targets] = two_people_map()
% 2PEOPLE_MAP Generates 2-people COVID gridworld map as matrix.

    % Map cells encoding values.
    free_val = 1;
    obstacle_val = 2;

    grid = ones(10, 10);
    grid = grid .* free_val;

    n = size(grid);

    % Vertical edges.
    for i = 1:n(1)
         grid(i, 1) = obstacle_val;
         grid(i, n(2)) = obstacle_val;
    end

    % Horizontal edges.
    for i = 1:n(2)
       grid(1, i) = obstacle_val;
       grid(n(1), i) = obstacle_val;
    end
    
    % Highest 1x2   
    for i = 4:5
         grid(2,i) = obstacle_val;
    end
    
    % Square 2x2
    for i = 5:6
        for j = 5:6
             grid(i,j) = obstacle_val;
        end
    end
        
    % Point
%     grid(6,4) = obstacle_val;
    
    
    
    % Generate targets
    targets = zeros(2,1);
    targets(1) = sub2ind([10 10], 6, 7);
    targets(2) = sub2ind([10 10], 2, 2);


end







