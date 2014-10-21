% Neural Network module
function [newWih, newWho, deltao] = backpropagation(valuesi, target, Wih, Who, rate)
    [valuesh, valueso, deltao] = feedfoward(Wih, Who, valuesi, target);
    [deltai, deltah] = backpropagate(Wih, Who, deltao);
    [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, deltai, deltah, deltao, rate);
endfunction

function [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, deltai, deltah, deltao, rate)
    % newWih = Wih + (rate * ((valuesi' * (valuesh .* (1 - valuesh))) * deltah))
    newWih = Wih + rate * (valuesi' * (deltah' .* (valuesh .* (1 - valuesh))));
    newWho = Who + rate * (valuesh' * (deltao' .* (valueso .* (1 - valueso))));
    % newWih = Wih + rate * (deltah * (valuesi' * (valuesh .* (1 - valuesh))))
    % newWho = Who + rate * (deltao * (valuesh' * (valueso .* (1 - valueso))))
endfunction

function [deltai, deltah] = backpropagate(Wih, Who, deltao)
    deltah = Who * deltao;
    deltai = Wih * deltah;
endfunction

function [valuesh, valueso, deltao] = feedfoward(Wih, Who, inputs, target)
   valuesh = arrayfun(@tanh, inputs * Wih);
   valueso = arrayfun(@tanh, valuesh * Who);
   deltao = ((target - valueso)' .^ 2) / 2;   % squared error
endfunction

% Activation function
% function y = sigmoid(x)
%     y = 1 / (1+e^(-x));
% endfunction
