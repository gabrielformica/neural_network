source('backpropagation.m')
arg_list = argv();
training_file = arg_list{1}
testing_file = arg_list{2}

hidden_nodes = arg_list{3}

function main(training_file, testing_file, hidden_nodes)
    [Wih, Who, biash, biaso] = initialize(2, str2num(hidden_nodes), 1);

    errors = [0];

    figure('name', strcat('errors -', training_file, '-', hidden_nodes));
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
        de = errors(i-1) - errors(i);
        if ((0 < de && de < 0.0000001 && i >= 1000) || err < 0.1)
            printf('exit by error at the %dth iteration\n', i);
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

    figure('name', strcat('test -', training_file, '-', hidden_nodes))
    plot(circsx,circsy) ;
    hold on;
    test(testing_file, Wih, Who, biash, biaso);
    hold off;
end

function test(testing_file, Wih, Who, biash, biaso)
    [xs,ys,ts] = textread(testing_file, '%f %f %f', 'delimiter', ' ');

    % outputs = [];
    for j = 1 : length(xs)
        if (ts(j) < 0)
            ts(j) = 0;
        end
        [valuesh, _, output, _, deltao] = feedforward(Wih, Who, biash, biaso, [xs(j), ys(j)], ts(j));
        % outputs(j) = output;

        if (round(output) == ts(j))
            if (round(output) == 0)
               plot(xs(j), ys(j), 'g+')
            else
               plot(xs(j), ys(j), 'r+')
            end
        else
            plot(xs(j), ys(j), 'k+')
        end
    end
end

function [Wih, Who, biash, biaso] = initialize(input_size, hidden_size, output_size)
    Wih = rand(input_size, hidden_size) - 0.5;
    Who = rand(hidden_size, output_size) - 0.5;
    biash = rand(1,hidden_size) - 0.5;
    biaso = rand(1,output_size) - 0.5;
end

main(training_file, testing_file, hidden_node)
