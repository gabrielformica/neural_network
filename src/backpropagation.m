% Neural Network module
1;

function [newWih, newWho, newBiash, newBiaso, deltao] = backpropagation(valuesi, target, Wih, Who, biash, biaso, rate)
    [valuesh, der_valuesh, valueso, der_valueso, deltao] = feedforward(Wih, Who, biash, biaso, valuesi, target);
    [deltah] = backpropagate(Wih, Who, deltao);
    [newWih, newWho, newBiash, newBiaso] = updateweights(Wih, Who, biash, biaso, valuesi, valuesh, der_valuesh, der_valueso, deltah, deltao, rate);
end

function [newWih, newWho, newBiash, newBiaso] = updateweights(Wih, Who, biash, biaso, valuesi, valuesh, der_valuesh, der_valueso, deltah, deltao, rate)
    newWih   = Wih + (rate .* (valuesi' * (deltah' .* der_valuesh)));
    newWho   = Who + (rate .* (valuesh' * (deltao' .* der_valueso)));
    newBiash = biash + (rate .* (deltah' .* der_valuesh));
    newBiaso = biaso + (rate .* (deltao' .* der_valueso));
end

function [deltah] = backpropagate(Wih, Who, deltao)
    deltah = Who * deltao;
    % deltai = Wih * deltah;
end

function [valuesh, der_valuesh, valueso, der_valueso, deltao] = feedforward(Wih, Who, biash, biaso, inputs, target)
    % tanh
    % valuesh = arrayfun(@tanh, (inputs * Wih) + biash);
    % valueso = arrayfun(@tanh, (valuesh * Who) + biaso);
    % der_valuesh = arrayfun(@sech, inputs * Wih) .^ 2;
    % der_valueso = arrayfun(@sech, valuesh * Who) .^ 2;

    % sigmoid'
    valuesh = arrayfun(@sigmoid, (inputs * Wih) + biash);
    valueso = arrayfun(@sigmoid, (valuesh * Who) + biaso);
    der_valuesh = valuesh .* (1 - valuesh);
    der_valueso = valueso .* (1 - valueso);

    deltao = (der_valueso .* (target - valueso))';
end

% Activation function
function y = sigmoid(x)
    y = 1 / (1+e^(-x));
end
