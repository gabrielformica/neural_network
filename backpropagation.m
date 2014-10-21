% Neural Network module
1;

function [newWih, newWho, deltao] = backpropagation(valuesi, target, Wih, Who, rate)
    [valuesh, der_valuesh, valueso, der_valueso, deltao] = feedforward(Wih, Who, valuesi, target);
    [deltai, deltah] = backpropagate(Wih, Who, deltao);
    [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, der_valuesh, der_valueso, deltai, deltah, deltao, rate);
end

function [newWih, newWho] = updateweights(Wih, Who, valuesi, valuesh, valueso, der_valuesh, der_valueso, deltai, deltah, deltao, rate)
    % deltaWih = valuesi' * (deltah' .* der_valuesh);
    % deltaWho = valuesh' * (deltao' .* der_valueso);

    Wih
    Who
    valuesi
    valuesh
    valueso
    der_valuesh
    der_valueso
    deltai
    deltah
    deltao
    rate

    deltaWih(1,1) = deltah(1) .* der_valuesh(1) .* valuesi(1);
    deltaWih(2,1) = deltah(1) .* der_valuesh(1) .* valuesi(2);

    deltaWih(1,2) = deltah(2) .* der_valuesh(2) .* valuesi(1);
    deltaWih(2,2) = deltah(2) .* der_valuesh(2) .* valuesi(2);

    deltaWih(1,3) = deltah(3) .* der_valuesh(3) .* valuesi(1);
    deltaWih(2,3) = deltah(3) .* der_valuesh(3) .* valuesi(2);

    deltaWho(1) = deltao .* der_valueso .* valuesh(1)
    deltaWho(2) = deltao .* der_valueso .* valuesh(2)
    deltaWho(3) = deltao .* der_valueso .* valuesh(3)

    error('parada');

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
   deltao = (target - valueso)';
end

% Activation function
function y = sigmoid(x)
    y = 1 / (1+e^(-x));
end
