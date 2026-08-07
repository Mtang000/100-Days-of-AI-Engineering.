import torch

fp32_weights = torch.tensor([
    [1.2345678, -2.9876543,  0.5555555],
    [-0.1111111,  3.1415926, -1.9999999]
], dtype=torch.float32)

print("Original Weights (32-bit Floats):")
print(fp32_weights)


max_val = torch.max(torch.abs(fp32_weights))
print(f"\nMax Absolute Value: {max_val:.4f}")


scale = 127.0 / max_val
print(f"Scaling Factor: {scale:.4f}")


int8_weights = torch.round(fp32_weights * scale).to(torch.int8)

print("\nQuantized Weights (8-bit Integers):")
print(int8_weights)

fp32_bytes = fp32_weights.nelement() * fp32_weights.element_size()
int8_bytes = int8_weights.nelement() * int8_weights.element_size()

print(f"\nFP32 Memory: {fp32_bytes} bytes")
print(f"INT8 Memory: {int8_bytes} bytes")


recovered_weights = (int8_weights.to(torch.float32) / scale)
print(recovered_weights)
