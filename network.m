% Wired values for debugging purposes
% Neural Network module
1;

function [Wih, Who] =  initialize(input_size, hidden_size, output_size)  
    Wih = ones(input_size, hidden_size);
    Who = ones(hidden_size, output_size);
   % Wih = rand(input_size, hidden_size);
   % Who = rand(hidden_size, output_size);
   % Signals.v1 = []    % input
   % Signals.v2 = []    % hidden
   % Signals.v3 = []    % output
endfunction 

function backpropagation
    [Wih, Who] = initialize(2,3,1);
    [valuesh, valueso, deltao] = feedfoward(Wih, Who, [2, 3], 1);   
    [deltai, deltah] = backpropagate(Wih, Who, deltao)
    updateweights(valuesh, valueso, deltai, deltah, deltao, 0.1)
endfunction

function updateweights(valuesh, valueso, deltai, deltah, deltao, rate) 
endfunction

function [deltai, deltah] = backpropagate(Wih, Who, deltao)
    deltah = Who * deltao' 
    deltai = Wih * deltah
endfunction

function [valuesh, valueso, deltao] = feedfoward(Wih, Who, inputs, target) 
   valuesh = arrayfun(@sigmoid, inputs * Wih)
   valueso = arrayfun(@sigmoid, valuesh * Who)
   deltao = target - valueso
endfunction

% Activation function
function y = sigmoid(x)
    y = 1 / (1+e^(-x));
endfunction
