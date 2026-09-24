import torch


def quantize_tensor(float_tensor, bit_width=8):
    q_min = -128
    q_max = 127

    val_min = float_tensor.min().item()
    val_max = float_tensor.max().item()

    scale = (val_max - val_min) / (q_max - q_min)

    zero_point = round(q_min - (val_min / scale))

    zero_point = max(q_min, min(q_max, zero_point))

    quantized_tensor = torch.round((float_tensor / scale) + zero_point)
    quantized_tensor = torch.clamp(quantized_tensor, q_min, q_max)

    quantized_tensor = quantized_tensor.to(torch.int8)

    return quantized_tensor, scale, zero_point


def dequantize_tensor(quantized_tensor, scale, zero_point):
    float_calc_tensor = quantized_tensor.to(torch.float32)

    dequantized_tensor = (float_calc_tensor - zero_point) * scale

    return dequantized_tensor


original_weights = torch.tensor(
    [-2.3451, -0.7182, 0.0000, 1.4592, 3.1415], dtype=torch.float32)
print(original_weights)

quantized_weights, scale, zero_point = quantize_tensor(original_weights)

print(f"\nCalculated Scale: {scale:.4f}")
print(f"Calculated Zero-Point: {zero_point}")

print(quantized_weights)
print(f"Data Type: {quantized_weights.dtype}")
dequantized_weights = dequantize_tensor(quantized_weights, scale, zero_point)

print(dequantized_weights)

error = torch.abs(original_weights - dequantized_weights)
print("\nRounding Error (Precision Loss):")
print(error)
