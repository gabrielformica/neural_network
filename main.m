source('backpropagation.m')

function main(training_file)
    [Wih, Who, biash, biaso] = initialize(2,8,1);

    errors = [0]

    figure;
    hold on;
    for i = 2 : 3000
        [xs,ys,ts] = textread(training_file, '%f %f %f', 'delimiter', ' ');
        err = 0;
        for j = 1:length(xs)
            if (ts(j) < 0)
                ts(j) = 0;
            end
            [Wih, Who, biash, biaso, delta] = backpropagation([xs(j), ys(j)], ts(j), Wih, Who, biash, biaso, 0.01);
            err = err + (delta .^ 2) ./ 2;
        end
        errors(i) = err;
        plot(i,err);
        if (abs(errors(i-1) - errors(i)) < 0.0001)
            disp('exit by error');
            break;
        end
    end
    err

    % plotting the circle
    radius  = 7;
    centerx = 10;
    centery = 10;
    t = linspace(0,2*pi,100)';
    circsx = radius .* cos(t) + centerx;
    circsy = radius .* sin(t) + centery;

    figure;
    plot(circsx,circsy);
    hold on;
    test(training_file, Wih, Who, biash, biaso);
    hold off;
end

function test(testing_file, Wih, Who, biash, biaso)
    [xs,ys,ts] = textread(testing_file, '%f %f %f', 'delimiter', ' ');

    outputs = [];
    for j = 1 : length(xs)
        if (ts(j) < 0)
            ts(j) = 0;
        end
        [valuesh, _, output, _, deltao] = feedforward(Wih, Who, biash, biaso, [xs(j), ys(j)], ts(j));
        outputs(j) = output;
        if (output >= 0.5)
           plot(xs(j), ys(j), 'g+')
        else
           plot(xs(j), ys(j), 'r+')
        end
    end
    outputs
end

function [Wih, Who, biash, biaso] = initialize(input_size, hidden_size, output_size)
    Wih = rand(input_size, hidden_size) - 0.5;
    Who = rand(hidden_size, output_size) - 0.5;
    biash = rand(1,hidden_size) - 0.5;
    biaso = rand(1,output_size) - 0.5;
end
