function main(training_file)
    [Wih, Who] = initialize(2,3,1);
    [xs,ys,ts] = textread(training_file, '%f %f %f', 'delimiter', ' ');
    for i = 1:length(xs)
        [Wih, Who] = backpropagation([xs(i), ys(i)], ts(i), Wih, Who, 0.1);
    endfor
    Wih
    Who
endfunction

function [Wih, Who] =  initialize(input_size, hidden_size, output_size)
   Wih = rand(input_size, hidden_size);
   Who = rand(hidden_size, output_size);
endfunction
