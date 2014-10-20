function [Wih, Who] =  initialize(input_size, hidden_size, output_size)
   Wih = rand(input_size, hidden_size);
   Who = rand(hidden_size, output_size);
endfunction
