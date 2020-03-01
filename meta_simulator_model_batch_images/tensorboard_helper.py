from tensorboardX import SummaryWriter
import os
class TensorBoardWriter(object):

    def __init__(self, folder, data_prefix):
        os.makedirs(folder,exist_ok=True)
        self.writer = SummaryWriter(folder)
        self.export_json_path = folder + "/all_scalars.json"
        self.data_prefix = data_prefix

    def record_trn_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_support_loss".format(self.data_prefix), tensor, iter)

    def record_trn_support_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_support_acc".format(self.data_prefix), tensor, iter)

    def record_trn_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_loss".format(self.data_prefix), tensor, iter)

    def record_trn_query_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_accuracy".format(self.data_prefix), tensor, iter)

    def record_trn_query_distance_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_distance_loss".format(self.data_prefix), tensor, iter)

    def record_trn_query_output_logits_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_output_logits_loss".format(self.data_prefix), tensor, iter)


    def export_json(self):
        self.writer.export_scalars_to_json(self.export_json_path)

    def close(self):
        self.writer.close()