1;
function [Wih, Who] =  initialize(input_size, hidden_size, output_size)
    Wih = ones(input_size, hidden_size)
    Who = ones(hidden_size, output_size)
    ones(10,10)
  % Wih = rand(input_size, hidden_size);
  % Who = rand(hidden_size, output_size);
endfunction
