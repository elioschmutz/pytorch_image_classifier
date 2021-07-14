from model_builder import ModelBuilder
import numpy as np
import os
import time
import torch


class ModelTrainer():

    def __init__(self, device=None, name="Model Trainer", checkpoint_dir='.', checkpoit_filename=None):
        self.set_device(device)
        self.best_trained_model = {}
        self.name = name
        self.checkpoint_file_path = self.generate_checkpoint_file_path(checkpoint_dir, checkpoit_filename)

    def __repr__(self):
        return(
            f"<ModelTrainer> {self.name} - "
            f"Is trained: {bool(self.best_trained_model)} - "
            f"Best model accuracy: {self.format_accuracy(self.get_best_trained_model_accuracy())}"
        )

    def set_model_builder(self, builder):
        self.model_builder = builder
        self.model, self.optimizer, self.criterion = self.model_builder.build()

        return self

    def train_model(self, dataloader_training, dataloader_validation, epochs=10):
        print(f'Running training with {epochs} epochs\n'
              f'Total training items: {len(dataloader_training.dataset)}\n'
              f'Total validation items: {len(dataloader_validation.dataset)}\n'
              f'Used device: {self.device}\n'
              f'Save checkpoint to: {self.checkpoint_file_path}')
        self.print_separator()

        train_losses = []
        validation_losses = []
        validation_accuracies = []

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}\n')
            loss_train, acc_train = self.process_loop(self.model, dataloader_training, self.criterion, self.optimizer)
            loss_validate, acc_validate = self.validate_model(dataloader_validation)

            print(f'Train loss: {loss_train:.3f}\n'
                  f'Validation loss: {loss_validate:.3f}\n'
                  f'Validation accuracy: {(acc_validate*100):.1f}\n'
                  )

            train_losses.append(loss_train)
            validation_losses.append(loss_validate)
            validation_accuracies.append(acc_validate)

            if self.get_best_trained_model_accuracy(fallback=-1) < acc_validate:
                print(f"Save new best trained model on epoch {epoch + 1}")
                self.set_best_trained_model(
                    epoch,
                    self.model,
                    self.optimizer,
                    loss_train,
                    acc_validate,
                    dataloader_training.dataset.class_to_idx)

            self.print_separator()

        print(f"Best trained model reached an accuracy of {(self.get_best_trained_model_accuracy() * 100):.1f}%")

    def validate_model(self, dataloader):
        with torch.no_grad():
            return self.process_loop(self.model, dataloader, self.criterion)

    def test_model(self, dataloader):
        print(f'Start testing model\n'
              f'Total test items: {len(dataloader.dataset)}\n'
              f'Used device: {self.device}')
        self.print_separator()

        with torch.no_grad():
            loss, acc = self.process_loop(self.model, dataloader)

        print(f'Test accuracy: {(acc*100):.1f}\n')

    def set_best_trained_model(self, epoch, model, optimizer, loss, accuracy, class_to_idx):
        self.best_trained_model = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "class_to_idx": class_to_idx
        }
        model.class_to_idx = class_to_idx
        self.save_best_model()

    def save_best_model(self):
        filepath = self.checkpoint_file_path
        torch.save({
            'best_model': self.best_trained_model,
            'model_builder': self.model_builder.builder_params
        }, filepath)

        print(f"Saved best model state to: {filepath}")
        return self

    def load_best_model(self):
        filepath = self.checkpoint_file_path
        self.print_separator()
        print(f"Loading best model from {filepath}...")

        self.load_model(filepath)

        return self

    def load_model(self, filepath=None, set_default_file_path=False):
        # https://github.com/pytorch/pytorch/issues/10622
        if self.device == 'cuda':
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        data = torch.load(filepath, map_location=map_location)
        
        if set_default_file_path:
            self.checkpoint_file_path = filepath

        best_model = data['best_model']
        model_builder = data['model_builder']

        self.best_trained_model = best_model
        self.set_model_builder(ModelBuilder().set_builder_params(model_builder))
        self.model.load_state_dict(best_model['model_state'])
        self.optimizer.load_state_dict(best_model['optimizer_state'])
        self.model.class_to_idx = best_model['class_to_idx']
        
        print(f"Loaded model: {self}")
        self.print_separator()

        return self

    def process_loop(self, model, dataloader, criterion=None, optimizer=None):
        if optimizer and not criterion:
            raise Exception("a criterion needs to be defined if an optimizer is provided")

        model.to(self.device)
        model.train() if optimizer else model.eval()

        running_loss = 0
        running_accuracy = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            if optimizer:
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        mean_accuracy = running_accuracy/len(dataloader)

        mean_loss = 0
        if criterion:
            mean_loss = running_loss/len(dataloader)

        return mean_loss, mean_accuracy

    def set_device(self, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return

        if device == 'cpu' or (device == 'cuda' and torch.cuda.is_available()):
            self.device = torch.device(device)
            return

        raise Exception(f"Device '{device}' is not supported")

    def get_best_trained_model_accuracy(self, fallback=0):
        return self.best_trained_model.get('accuracy', fallback)

    def print_separator(self):
        print('-'*50)

    def format_accuracy(self, value):
        if not value:
            value = 0
        return f"{value*100:.1f}%"

    def generate_checkpoint_file_path(self, checkpoint_dir='.', filename=None):
        filename = filename or self.generate_random_checkpoint_filename()
        return os.path.join(checkpoint_dir, filename)

    def generate_random_checkpoint_filename(self):
        return f"{self.name.lower().replace(' ', '_')}_{round(time.time() * 1000)}.pth"
