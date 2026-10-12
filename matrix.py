"""
=====================================================================
PENUGASAN KOMPUTASI NUMERIK
Penyelesaian Sistem Persamaan Linear (SPL) 3 Variabel
dengan Metode Jacobi dan Metode Gauss-Seidel
Studi Kasus: Harga Buah di Toko Buah (Apel, Jeruk, Mangga)
=====================================================================

Sistem persamaan:
    4x +  y +  z = 14000   (Persamaan 1)
     x + 5y +  z = 15000   (Persamaan 2)
     x +  y + 6z = 21000   (Persamaan 3)

dengan x = harga apel/kg, y = harga jeruk/kg, z = harga mangga/kg.
"""

# Mengimpor modul "math" untuk keperluan matematika dasar (opsional, disiapkan bila diperlukan)
import math


# ---------------------------------------------------------------------
# FUNGSI BANTU 1: Memeriksa syarat diagonally dominant
# ---------------------------------------------------------------------
def cek_diagonal_dominan(A):
    """
    Memeriksa apakah matriks A bersifat strictly diagonally dominant,
    yaitu |a_ii| > jumlah |a_ij| untuk semua j != i, pada setiap baris i.
    Syarat ini menjamin Jacobi dan Gauss-Seidel konvergen.
    """
    # Menentukan jumlah baris/ukuran matriks A
    n = len(A)
    # Perulangan untuk memeriksa setiap baris matriks
    for i in range(n):
        # Mengambil nilai mutlak elemen diagonal pada baris ke-i
        diagonal = abs(A[i][i])
        # Menjumlahkan nilai mutlak semua elemen non-diagonal pada baris ke-i
        jumlah_lain = sum(abs(A[i][j]) for j in range(n) if j != i)
        # Jika diagonal tidak lebih besar dari jumlah elemen lain, syarat gagal
        if diagonal <= jumlah_lain:
            # Mengembalikan False karena baris ini melanggar syarat
            return False
    # Jika semua baris lolos pemeriksaan, kembalikan True
    return True


# ---------------------------------------------------------------------
# FUNGSI BANTU 2: Mencetak satu baris tabel iterasi
# ---------------------------------------------------------------------
def cetak_baris(k, x, galat):
    """Mencetak satu baris log iterasi: nomor iterasi, nilai x, y, z, dan galat."""
    # Memformat dan mencetak baris dengan lebar kolom tetap agar rapi
    print(f"{k:>5} | {x[0]:>14.4f} | {x[1]:>14.4f} | {x[2]:>14.4f} | {galat:>12.6f}")


# ---------------------------------------------------------------------
# FUNGSI 1: METODE JACOBI
# ---------------------------------------------------------------------
def metode_jacobi(A, b, x0, toleransi=1e-4, maks_iterasi=50, tampilkan=True):
    """
    Menyelesaikan A.x = b dengan Metode Jacobi.

    Cara kerja: pada iterasi ke-(k+1), SEMUA nilai baru dihitung hanya
    dari nilai iterasi ke-k (nilai lama). Nilai lama disimpan utuh
    selama satu putaran penuh, sehingga seluruh variabel diperbarui
    secara serentak (simultan).

    Parameter:
        A            : matriks koefisien (list of list)
        b            : vektor ruas kanan (list)
        x0           : tebakan awal (list)
        toleransi    : batas galat untuk berhenti
        maks_iterasi : batas jumlah iterasi
        tampilkan    : True bila log iterasi ingin dicetak

    Mengembalikan: (solusi, jumlah_iterasi, konvergen)
    """
    # Menentukan jumlah variabel (ukuran sistem)
    n = len(b)
    # Menyalin tebakan awal agar list asli x0 tidak ikut berubah
    x_lama = x0[:]

    # Mencetak judul dan header tabel bila log diminta
    if tampilkan:
        print("\n" + "=" * 70)
        print("METODE JACOBI")
        print("=" * 70)
        print(f"{'Iter':>5} | {'x (Apel)':>14} | {'y (Jeruk)':>14} | {'z (Mangga)':>14} | {'Galat':>12}")
        print("-" * 70)
        # Mencetak kondisi awal (iterasi ke-0) dengan galat tidak terdefinisi (0)
        cetak_baris(0, x_lama, 0.0)

    # Perulangan utama iterasi, dimulai dari 1 sampai maks_iterasi
    for k in range(1, maks_iterasi + 1):
        # Menyiapkan list KOSONG untuk menampung nilai baru (x_baru terpisah dari x_lama)
        x_baru = [0.0] * n

        # Perulangan untuk menghitung setiap variabel ke-i
        for i in range(n):
            # Variabel penampung jumlah suku a_ij * x_j untuk j != i
            jumlah = 0.0
            # Perulangan atas semua kolom j
            for j in range(n):
                # Melewati suku diagonal karena variabel x_i yang sedang dicari
                if j != i:
                    # Menjumlahkan a_ij * x_j memakai nilai LAMA (x_lama), bukan nilai baru
                    jumlah += A[i][j] * x_lama[j]
            # Menghitung x_i baru = (b_i - jumlah) / a_ii, disimpan di x_baru
            x_baru[i] = (b[i] - jumlah) / A[i][i]

        # Menghitung galat = selisih mutlak terbesar antara nilai baru dan lama (norma tak hingga)
        galat = max(abs(x_baru[i] - x_lama[i]) for i in range(n))

        # Mencetak hasil iterasi ke-k bila log diminta
        if tampilkan:
            cetak_baris(k, x_baru, galat)

        # Mengecek kriteria berhenti: galat sudah lebih kecil dari toleransi
        if galat < toleransi:
            # Mengembalikan solusi, jumlah iterasi, dan status konvergen = True
            return x_baru, k, True

        # Setelah satu putaran selesai, nilai baru menjadi nilai lama untuk iterasi berikutnya
        x_lama = x_baru

    # Bila loop habis tanpa memenuhi toleransi, kembalikan hasil terakhir dengan status False
    return x_lama, maks_iterasi, False


# ---------------------------------------------------------------------
# FUNGSI 2: METODE GAUSS-SEIDEL
# ---------------------------------------------------------------------
def metode_gauss_seidel(A, b, x0, toleransi=1e-4, maks_iterasi=50, tampilkan=True):
    """
    Menyelesaikan A.x = b dengan Metode Gauss-Seidel.

    Cara kerja: nilai yang baru dihitung LANGSUNG dipakai pada
    perhitungan variabel berikutnya di iterasi yang sama (in-place
    update). Misalnya, x yang baru langsung dipakai untuk menghitung y,
    lalu x dan y yang baru dipakai untuk menghitung z.

    Parameter dan keluaran sama dengan metode_jacobi.
    """
    # Menentukan jumlah variabel (ukuran sistem)
    n = len(b)
    # Menyalin tebakan awal; list ini akan DIPERBARUI LANGSUNG (in-place) tiap variabel
    x = x0[:]

    # Mencetak judul dan header tabel bila log diminta
    if tampilkan:
        print("\n" + "=" * 70)
        print("METODE GAUSS-SEIDEL")
        print("=" * 70)
        print(f"{'Iter':>5} | {'x (Apel)':>14} | {'y (Jeruk)':>14} | {'z (Mangga)':>14} | {'Galat':>12}")
        print("-" * 70)
        # Mencetak kondisi awal (iterasi ke-0)
        cetak_baris(0, x, 0.0)

    # Perulangan utama iterasi
    for k in range(1, maks_iterasi + 1):
        # Menyimpan salinan nilai sebelum iterasi, hanya untuk menghitung galat
        x_sebelum = x[:]

        # Perulangan untuk memperbarui setiap variabel ke-i secara berurutan
        for i in range(n):
            # Variabel penampung jumlah suku a_ij * x_j untuk j != i
            jumlah = 0.0
            # Perulangan atas semua kolom j
            for j in range(n):
                # Melewati suku diagonal
                if j != i:
                    # Memakai x[j] TERKINI: untuk j < i sudah nilai baru, untuk j > i masih nilai lama
                    jumlah += A[i][j] * x[j]
            # Menghitung x_i baru dan MENIMPA langsung x[i] (in-place update)
            x[i] = (b[i] - jumlah) / A[i][i]

        # Menghitung galat = selisih mutlak terbesar antara nilai sekarang dan sebelum iterasi
        galat = max(abs(x[i] - x_sebelum[i]) for i in range(n))

        # Mencetak hasil iterasi ke-k bila log diminta
        if tampilkan:
            cetak_baris(k, x, galat)

        # Mengecek kriteria berhenti
        if galat < toleransi:
            # Mengembalikan salinan solusi, jumlah iterasi, dan status konvergen
            return x[:], k, True

    # Bila batas iterasi tercapai tanpa konvergen
    return x[:], maks_iterasi, False


# ---------------------------------------------------------------------
# BLOK EKSEKUSI UTAMA
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Matriks koefisien A dari tiga persamaan toko buah
    A = [
        [4.0, 1.0, 1.0],   # Persamaan 1: 4x +  y +  z
        [1.0, 5.0, 1.0],   # Persamaan 2:  x + 5y +  z
        [1.0, 1.0, 6.0],   # Persamaan 3:  x +  y + 6z
    ]
    # Vektor ruas kanan b (total belanja dalam Rupiah)
    b = [14000.0, 15000.0, 21000.0]
    # Tebakan awal [x0, y0, z0] = [0, 0, 0]
    x_awal = [0.0, 0.0, 0.0]
    # Toleransi galat sebesar 10^-4
    toleransi = 1e-4
    # Batas maksimum iterasi
    maks_iterasi = 50

    # Mencetak ringkasan parameter simulasi
    print("PARAMETER SIMULASI")
    print(f"Tebakan awal  : {x_awal}")
    print(f"Toleransi     : {toleransi}")
    print(f"Maks iterasi  : {maks_iterasi}")
    # Memverifikasi syarat diagonally dominant dan mencetak statusnya
    print(f"Diagonally dominant: {'YA (konvergen dijamin)' if cek_diagonal_dominan(A) else 'TIDAK'}")

    # Menjalankan Metode Jacobi dan menampung hasilnya
    sol_j, iter_j, ok_j = metode_jacobi(A, b, x_awal, toleransi, maks_iterasi)
    # Menjalankan Metode Gauss-Seidel dan menampung hasilnya
    sol_gs, iter_gs, ok_gs = metode_gauss_seidel(A, b, x_awal, toleransi, maks_iterasi)

    # Mencetak ringkasan hasil akhir kedua metode
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL AKHIR")
    print("=" * 70)
    # Membuat daftar (nama, solusi, iterasi, status) agar pencetakan ringkas
    for nama, sol, it, ok in [("Jacobi", sol_j, iter_j, ok_j),
                              ("Gauss-Seidel", sol_gs, iter_gs, ok_gs)]:
        # Menentukan teks status konvergensi
        status = "konvergen" if ok else "TIDAK konvergen"
        print(f"\nMetode {nama} ({it} iterasi, {status}):")
        print(f"  Harga Apel   (x) = Rp {sol[0]:,.2f} per kg")
        print(f"  Harga Jeruk  (y) = Rp {sol[1]:,.2f} per kg")
        print(f"  Harga Mangga (z) = Rp {sol[2]:,.2f} per kg")

    # Menampilkan perbandingan jumlah iterasi
    print(f"\nPerbandingan: Jacobi = {iter_j} iterasi, Gauss-Seidel = {iter_gs} iterasi")