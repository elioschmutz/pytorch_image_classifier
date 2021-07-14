from data_loader import get_loader
from model_builder import ModelBuilder
from model_trainer import ModelTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Image model trainer')
    parser.add_argument('data_directory', type=str)
    parser.add_argument('--name', type=str, default='Model Trainer')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg11', 'vgg16'])
    parser.add_argument('--hidden_units', type=int, default=[], nargs="*")
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--filename', type=str, help="If no filename is given, a random name will be used")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])

    args = parser.parse_args()
    train_loader = get_loader('train', args.data_directory)
    valid_loader = get_loader('valid', args.data_directory)
    model_builder = ModelBuilder()
    model_builder.with_model(args.arch, len(train_loader.dataset.class_to_idx), args.dropout, args.hidden_units)
    model_builder.with_optimizer('sgd', momentum=args.momentum, lr=args.learning_rate)
    model_builder.with_criterion('nllloss')

    trainer = ModelTrainer(device=args.device,
                           name=args.name,
                           checkpoint_dir=args.save_dir,
                           checkpoit_filename=args.filename)
    trainer.set_model_builder(model_builder)
    trainer.train_model(train_loader, valid_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
