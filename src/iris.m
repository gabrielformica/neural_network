source('backpropagation.m')
arg_list = argv();
training_file = arg_list{1}
testing_file = arg_list{2}

hidden_nodes = arg_list{3}

function main(training_file, testing_file, hidden_nodes)
    [Wih, Who, biash, biaso] = initialize(4, str2num(hidden_nodes), 3);

    errors = [0];

    figure('name', strcat('errors -', training_file, '-', hidden_nodes));
    hold on;

    [sls,sws,pls,pws,cls] = textread(training_file, '%f %f %f %f %d', 'delimiter', ' ');
    features = [sls,sws,pls,pws];

    for i = 2 : 3000
        err = 0;
        for j = 1:length(features(:,1))
            if (cls(j) == 0)        % Iris-setosa
                target = [1,0,0];
            elseif (cls(j) == 1)    % Iris-versicolor
                target = [0,1,0];
            elseif (cls(j) == 2)    % Iris-virginica
                target = [0,0,1];
            else
                error('unknown class');
            end
            [Wih, Who, biash, biaso, delta] = backpropagation(features(j,:), target, Wih, Who, biash, biaso, 0.01);
            for k = 1:length(delta)
                err = err .+ (delta(k) .^ 2) ./ 2;
            end
        end
        errors(i) = err;
        plot(i,err);
        de = errors(i-1) - errors(i);
        if ((0 < de && de < 0.0001 && i >= 1500) || err < 0.1)
            printf('exit by error at the %dth iteration\n', i);
            break;
        end
    end
    err

    test(testing_file, Wih, Who, biash, biaso);
end

function test(testing_file, Wih, Who, biash, biaso)
    [sls,sws,pls,pws,cls] = textread(testing_file, '%f %f %f %f %d', 'delimiter', ' ');
    features = [sls,sws,pls,pws];

    right = 0;

    % outputs = [];
    for j = 1:length(features(:,1))
        if (cls(j) == 0)        % Iris-setosa
            target = [1,0,0];
        elseif (cls(j) == 1)    % Iris-versicolor
            target = [0,1,0];
        elseif (cls(j) == 2)    % Iris-virginica
            target = [0,0,1];
        else
            error('unknown class');
        end
        [_, _, output, _, _] = feedforward(Wih, Who, biash, biaso, features(j,:), target);

        if (round(output) == target)
            right += 1;
        end
    end

    printf('%d right guesses out of %d.', right, length(features(:,1)));
end

function [Wih, Who, biash, biaso] = initialize(input_size, hidden_size, output_size)
    Wih = rand(input_size, hidden_size) - 0.5;
    Who = rand(hidden_size, output_size) - 0.5;
    biash = rand(1,hidden_size) - 0.5;
    biaso = rand(1,output_size) - 0.5;
end

main(training_file, testing_file, hidden_nodes)
