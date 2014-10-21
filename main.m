function main(training_file)
    [Wih, Who] = initialize(2,3,1);
    [xs,ys,ts] = textread(training_file, '%f %f %f', 'delimiter', ' ');
    deltas = [];
    % for i = 1:100
    %     delta = 0;
    %     for x = 1:length(xs)
    %         [Wih, Who, deltao] = backpropagation([xs(x), ys(x)], ts(x), Wih, Who, 0.1);
    %         delta = delta + deltao;
    %     endfor
    %     deltas(i) = delta;
    % endfor
    for i = 0:999
        for x = 1:length(xs)
            [Wih, Who, deltao] = backpropagation([xs(x), ys(x)], ts(x), Wih, Who, 0.1);
            deltas(x + i * length(xs)) = deltao;
        endfor
    endfor
    figure
    plot(deltas, 'markersize', 2, '*');
    Wih
    Who
endfunction

function [Wih, Who] =  initialize(input_size, hidden_size, output_size)
   Wih = rand(input_size, hidden_size);
   Who = rand(hidden_size, output_size);
endfunction
