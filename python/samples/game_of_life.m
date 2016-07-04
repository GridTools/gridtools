%% Multi Game of Life
% This is an example script running a Game of Life on multiple domains
% The code is based on Chapter 12 of Cleve Moler's "Experiments with MATLAB",
% available in electronic form at http://www.mathworks.com/moler/exm/index.html

% Generate a random initial population
X = sparse(50,50,4);
X(19:32,19:32,:) = (rand(14,14,4) > .75);
p0 = nnz(X);

% Whether cells stay alive, die, or generate new cells depends
% upon how many of their eight possible neighbors are alive.
% Index vectors increase or decrease the centered index by one.
n = size(X,1);
p = [1 1:n-1];
q = [2:n n];

% Loop over 100 generations.
for t = 1:100

    %spy(X(:,:,1))
    %title(num2str(t))
    %drawnow

    % Count how many of the eight neighbors are alive.
    Y = X(:,p,:) + X(:,q,:) + X(p,:,:) + X(q,:,:) + ...
        X(p,p,:) + X(q,q,:) + X(p,q,:) + X(q,p,:);

    % A live cell with two live neighbors, or any cell with
    % three live neighbors, is alive at the next step.
    X = (X & (Y == 2)) | (Y == 3);
end
