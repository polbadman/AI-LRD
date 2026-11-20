<?php
// ============================================
//  تنظیمات قابل تغییر
// ============================================
$CONFIG = [
    'hidden_size' => 4,      // تعداد نرون مخفی (۲ تا ۸ رو امتحان کن)
    'learning_rate' => 0.5,  // نرخ یادگیری
    'epochs' => 20000,       // تعداد دورهای آموزش
    'seed' => 42,            // برای تکرارپذیری
    'csv_file' => 'xor_training.csv', // فایل خروجی خطا
];

// ===== توابع =====
function sigmoid($x) { return 1 / (1 + exp(-$x)); }
function sigmoid_derivative($out) { return $out * (1 - $out); }

// ===== داده‌ها =====
$inputs  = [[0,0],[0,1],[1,0],[1,1]];
$targets = [0, 1, 1, 0];

$inputSize  = 2;
$hiddenSize = $CONFIG['hidden_size'];
$outputSize = 1;
$lr         = $CONFIG['learning_rate'];
$epochs     = $CONFIG['epochs'];

mt_srand($CONFIG['seed']);

// ===== مقداردهی وزن‌ها =====
$W1 = [];
for ($h = 0; $h < $hiddenSize; $h++)
    for ($i = 0; $i < $inputSize; $i++)
        $W1[$h][$i] = (mt_rand() / mt_getrandmax()) * 2 - 1;

$b1 = array_fill(0, $hiddenSize, 0.0);

$W2 = [];
for ($o = 0; $o < $outputSize; $o++)
    for ($h = 0; $h < $hiddenSize; $h++)
        $W2[$o][$h] = (mt_rand() / mt_getrandmax()) * 2 - 1;

$b2 = array_fill(0, $outputSize, 0.0);

// ===== فایل CSV =====
$csv = fopen($CONFIG['csv_file'], 'w');
fputcsv($csv, ['epoch', 'total_error']);

// ===== آموزش =====
echo "=== آموزش با {$hiddenSize} نرون مخفی، lr={$lr} ===\n";

for ($epoch = 0; $epoch < $epochs; $epoch++) {
    $totalError = 0;

    foreach ($inputs as $idx => $x) {
        $target = $targets[$idx];

        // --- Forward ---
        $hiddenOut = [];
        for ($h = 0; $h < $hiddenSize; $h++) {
            $sum = $b1[$h];
            for ($i = 0; $i < $inputSize; $i++)
                $sum += $W1[$h][$i] * $x[$i];
            $hiddenOut[$h] = sigmoid($sum);
        }

        $outputOut = [];
        for ($o = 0; $o < $outputSize; $o++) {
            $sum = $b2[$o];
            for ($h = 0; $h < $hiddenSize; $h++)
                $sum += $W2[$o][$h] * $hiddenOut[$h];
            $outputOut[$o] = sigmoid($sum);
        }

        // --- خطا ---
        $error = $target - $outputOut[0];
        $totalError += $error * $error;

        // --- Backward ---
        $dOutput = $error * sigmoid_derivative($outputOut[0]);

        $dHidden = [];
        for ($h = 0; $h < $hiddenSize; $h++) {
            $sum = 0;
            for ($o = 0; $o < $outputSize; $o++)
                $sum += $dOutput * $W2[$o][$h];
            $dHidden[$h] = $sum * sigmoid_derivative($hiddenOut[$h]);
        }

        // --- به‌روزرسانی ---
        for ($o = 0; $o < $outputSize; $o++) {
            for ($h = 0; $h < $hiddenSize; $h++)
                $W2[$o][$h] += $lr * $dOutput * $hiddenOut[$h];
            $b2[$o] += $lr * $dOutput;
        }

        for ($h = 0; $h < $hiddenSize; $h++) {
            for ($i = 0; $i < $inputSize; $i++)
                $W1[$h][$i] += $lr * $dHidden[$h] * $x[$i];
            $b1[$h] += $lr * $dHidden[$h];
        }
    }

    // ذخیره خطا در CSV (هر دور)
    fputcsv($csv, [$epoch, $totalError]);

    if ($epoch % 2000 == 0)
        echo "Epoch $epoch | Error: " . round($totalError, 6) . "\n";
}
fclose($csv);

// ===== تست نهایی =====
echo "\n===== نتیجه نهایی =====\n";
foreach ($inputs as $idx => $x) {
    $hiddenOut = [];
    for ($h = 0; $h < $hiddenSize; $h++) {
        $sum = $b1[$h];
        for ($i = 0; $i < $inputSize; $i++)
            $sum += $W1[$h][$i] * $x[$i];
        $hiddenOut[$h] = sigmoid($sum);
    }
    $sum = $b2[0];
    for ($h = 0; $h < $hiddenSize; $h++)
        $sum += $W2[0][$h] * $hiddenOut[$h];
    $out = sigmoid($sum);

    printf("XOR(%d, %d) = %.4f  (انتظار: %d)\n",
        $x[0], $x[1], $out, $targets[$idx]);
}