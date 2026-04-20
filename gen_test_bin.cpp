#include <stdio.h>
#include "kuiper/include/model/config.h"

int main() {
    FILE* f = fopen("tmp/test.bin", "wb");
    if (!f) return 1;
    model::ModelConfig config;
    config.dim = 16;
    config.hidden_dim = 128;
    config.layer_num = 256;
    fwrite(&config, sizeof(config), 1, f);
    for (int i = 0; i < 16 * 128; ++i) {
        float val = i;
        fwrite(&val, sizeof(float), 1, f);
    }
    fclose(f);
    return 0;
}
