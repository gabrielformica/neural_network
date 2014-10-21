% Neural Network module
1;

function [newWih, newWho] = backpropagation(valuesi, target, Wih, Who, rate)
    [valuesh, valueso, deltao] = feedforward(Wih, Who, valuesi, target);
    [deltai, deltah] = backpropagate(Wih, Who, deltao);
    [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, deltai, deltah, deltao, rate);
endfunction

function [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, deltai, deltah, deltao, rate)
    % newWih = Wih + (rate * ((valuesi' * (valuesh .* (1 - valuesh))) * deltah))
    deltaWih = rate * (valuesi' * (deltah' .* (valuesh .* (1 - valuesh))));
    newWih = Wih + deltaWih;
    deltaWho = rate * (valuesh' * (deltao' .* (valueso .* (1 - valueso))));
    newWho = Who + deltaWho;
    % newWih = Wih + rate * (deltah * (valuesi' * (valuesh .* (1 - valuesh))))
    % newWho = Who + rate * (deltao * (valuesh' * (valueso .* (1 - valueso))))
endfunction

function [deltai, deltah] = backpropagate(Wih, Who, deltao)
    deltah = Who * deltao;
    deltai = Wih * deltah;
endfunction

function [valuesh, valueso, deltao] = feedforward(Wih, Who, inputs, target)
   valuesh = arrayfun(@sigmoid, inputs * Wih);
   valueso = arrayfun(@sigmoid, valuesh * Who);
   deltao = (target - valueso)';
endfunction

% Activation function
function y = sigmoid(x)
    y = 1 / (1+e^(-x));
endfunction
