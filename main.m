source('backpropagation.m')

function main(training_file)
    [Wih, Who] = initialize(2,3,1)
    figure;

    for j = 1 : 100
       [xs,ys,ts] = textread(training_file, '%f %f %f', 'delimiter', ' ');
       for i = 1:length(xs)
           [Wih, Who] = backpropagation([xs(i), ys(i)], ts(i), Wih, Who, 0.01);
       end
    end

    % plotting the circle
    radius  = 7;
    centerx = 10;
    centery = 10;
    t = linspace(0,2*pi,100)';
    circsx = radius .* cos(t) + centerx;
    circsy = radius .* sin(t) + centery;
    plot(circsx,circsy);

    hold on;
    test(training_file, Wih, Who);
    hold off;
end

function test(testing_file, Wih, Who)
    [xs,ys,ts] = textread(testing_file, '%f %f %f', 'delimiter', ' ');

    for i = 1 : length(xs)
        [valuesh, _, output, _, deltao] = feedforward(Wih, Who, [xs(i), ys(i)], ts(i));
        if (output >= 0)
           plot(xs(i), ys(i), 'g+')
        else
           plot(xs(i), ys(i), 'r+')
        end
    end
end

function [Wih, Who] =  initialize(input_size, hidden_size, output_size)
    % Wih = rand(input_size, hidden_size) - 0.5;
    % Who = rand(hidden_size, output_size) - 0.5;
    Wih = ones(input_size, hidden_size);
    Who = ones(hidden_size, output_size);
endfunction
