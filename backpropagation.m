% Neural Network module
1;

function [newWih, newWho, deltao] = backpropagation(valuesi, target, Wih, Who, rate)
    [valuesh, der_valuesh, valueso, der_valueso, deltao] = feedforward(Wih, Who, valuesi, target);
    [deltai, deltah] = backpropagate(Wih, Who, deltao);
    [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, der_valuesh, der_valueso, deltai, deltah, deltao, rate);
end

function [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, der_valuesh, der_valueso, deltai, deltah, deltao, rate)
    deltaWih = valuesi' * (deltah' .* der_valuesh);
    deltaWho = valuesh' * (deltao' .* der_valueso);

    newWih   = Wih + (rate .* deltaWih);
    newWho   = Who + (rate .* deltaWho);
end

function [deltai, deltah] = backpropagate(Wih, Who, deltao)
    deltah = Who * deltao;
    deltai = Wih * deltah;
end

function [valuesh, der_valuesh, valueso, der_valueso, deltao] = feedforward(Wih, Who, inputs, target)
  % tanh
   % valuesh = arrayfun(@tanh, inputs * Wih);
   % valueso = arrayfun(@tanh, valuesh * Who);
   % der_valuesh = arrayfun(@sech, inputs * Wih) .^ 2;
   % der_valueso = arrayfun(@sech, valuesh * Who) .^ 2;

   % sigmoid'
   valuesh = arrayfun(@sigmoid, inputs * Wih);
   valueso = arrayfun(@sigmoid, valuesh * Who);
   der_valuesh = valuesh .* (1 - valuesh);
   der_valueso = valueso .* (1 - valueso);

   % squared error
   % deltao = ((target - valueso)' .^ 2) ./ 2;
   deltao = (valueso) .* (1 - valueso) .* (target - valueso)'
end

% Activation function
function y = sigmoid(x)
    y = 1 / (1+e^(-x));
end
