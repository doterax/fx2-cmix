// ppmd is written by Dmitry Shkarin.
// mod_ppmd is adapted from ppmd by Eugene Shelwien.
// This file is adapted from mod_ppmd_v2: http://encode.su/threads/2515-mod_ppmd
// 15-Nov-2025: Alex.A.Yermoshenko (with help from ChatGPT):
//               fix crush in reclaiming/pruning context memory blocks
//               now it works correctly with all range of memory sizes

//
// PPMD (Prediction by Partial Matching, variant D) is a sophisticated
// statistical compression algorithm that predicts the next byte based on
// context.
//
// Key components:
// 1. Custom Memory Allocator - Efficiently manages millions of small
// allocations
// 2. Context Tree - Stores symbol statistics up to order 25
// 3. Prediction Engine - Generates probabilities with escape mechanism
// 4. Adaptive Learning - Updates frequencies as data is processed
//
// See PPMD.md for detailed documentation.

#include "ppmd.h"

#include <cassert>


/*
PPMD Heap Memory Layout
===============================================================================

Total Size: SubAllocatorSize (e.g., 1 GB = 1,048,576 KB = 1,073,741,824 bytes)
Unit Size: UNIT_SIZE = 12 bytes (stores one PPM_CONTEXT)

===============================================================================
ADDRESS         REGION              GROWS       DESCRIPTION
===============================================================================
HeapStart   →   ┌─────────────────┐
                │                 │
                │   Text Area     │    ↓       Temporary storage for context
                │   (pText)       │  (down)    data. Used during model
                │                 │            operations. Expands downward
                │                 │            from HeapStart.
pText       →   ├─────────────────┤            Size: (pText - HeapStart)
                │                 │
                │   Free Space    │            Gap between text area and units
                │                 │
UnitsStart  →   ├─────────────────┤  ←───────  1/8 of heap reserved for text
                │                 │            area. 7/8 of heap for context
                │  Free List Area │            storage.
                │  (BList[])      │            Segregated free lists by size
                │                 │            N_INDEXES=38 buckets
                │  ╔═════════════╗│
                │  ║ Free Blocks ║│            Recycled memory blocks
                │  ╚═════════════╝│            Organized by unit count
LoUnit      →   ├─────────────────┤
                │                 │    ↑
                │   Allocated     │  (up)      STATE arrays & multi-stat
                │   Units         │            contexts Bump allocated from
                │   (Data)        │            LoUnit upward. Each unit = 12
                │                 │            bytes. Grows toward HiUnit
                ├─────────────────┤
                │   Free Space    │            Available memory between
                │                 │            allocators. When
HiUnit      →   ├─────────────────┤            LoUnit == HiUnit → exhausted
                │                 │    ↓
                │   Allocated     │  (down)    PPM_CONTEXT nodes
                │   Contexts      │            Bump allocated from HiUnit
                │   (AllocContext)│            downward Each context = 12 bytes
                │                 │            (1 UNIT) Grows toward LoUnit
HeapStart +     └─────────────────┘
SubAllocatorSize

===============================================================================

KEY POINTERS:
-------------
HeapStart    : Base address of entire heap (never changes)
pText        : End of text area (grows down from HeapStart)
UnitsStart   : Boundary between text area (1/8) and context storage (7/8)
               Can move forward during ExpandTextArea() - INVALIDATES all
               pointers!
LoUnit       : Bottom of allocated units region (grows up)
HiUnit       : Top of allocated contexts region (grows down)
AuxUnit      : Auxiliary unit for text area operations

MEMORY CALCULATION:
-------------------
Total Heap:      SubAllocatorSize bytes
Text Area:       1/8 of heap = SubAllocatorSize / 8
Context Storage: 7/8 of heap = (SubAllocatorSize / 8 / 12) * 7 * 12 bytes
Free Space:      HiUnit - LoUnit
Used Memory:     SubAllocatorSize - (HiUnit - LoUnit) - (UnitsStart - pText)

ALLOCATION STRATEGIES:
----------------------
1. AllocContext():    Allocates from HiUnit downward (fast bump allocation)
                      Used for PPM_CONTEXT structures (12 bytes each)

2. AllocUnits(NU):    Allocates NU units from LoUnit upward
                      Used for STATE arrays (6 bytes each, 2 per unit)
                      Try free list → bump allocate → defragment
                      (AllocUnitsRare)

3. Free List (BList): 38 buckets for different unit counts
                      Segregated fit allocation for fast reuse
                      Reduces fragmentation

CRITICAL ISSUES:
----------------
⚠️  After ExpandTextArea(), UnitsStart moves forward
    → All context POINTERS become invalid (point to old heap locations)
    → Only INDICES (iStats, iSuffix, iSuccessor) remain valid
    → Must convert indices back to pointers using Indx2Ptr() after expansion

⚠️  When LoUnit == HiUnit: Heap exhausted
    → Trigger RestoreModelRare() or StartModelRare()
    → InitSubAllocator() resets pText, UnitsStart, LoUnit, HiUnit

STRUCTURE SIZES:
----------------
STATE:        6 bytes  (Symbol:1, Freq:1, iSuccessor:4)
PPM_CONTEXT:  12 bytes (NumStats:1, Flags:1, SummFreq:2, iStats:4, iSuffix:4)
UNIT_SIZE:    12 bytes (stores 1 context or 2 states)
MEM_BLK:      varies   (header for free blocks with NU field)

===============================================================================
*/

namespace PPMD {

template <class T> T Min(T x, T y) {
  return (x < y) ? x : y;
}
template <class T> T Max(T x, T y) {
  return (x > y) ? x : y;
}

#pragma pack(1)

typedef unsigned short     word;
typedef unsigned int       uint;
typedef unsigned char      byte;
typedef unsigned long long qword;

const int                  ORealMAX      = 256;

static signed char         EscCoef[12]   = {16, -10, 1,  51, 14,  89,
                                            23, 35,  64, 26, -42, 43};

static const byte          ExpEscape[16] = {51, 43, 18, 12, 11, 9, 8, 7,
                                            6,  5,  4,  3,  3,  2, 2, 2};

// Main PPMD model structure containing:
// - Custom memory allocator for efficient context management
// - Context tree data structures
// - Prediction and update algorithms
struct ppmd_Model {

  typedef unsigned short     word;
  typedef unsigned int       uint;
  typedef unsigned char      byte;
  typedef unsigned long long qword;

  enum { SCALE = 1 << 15 };

  // Memory thresholds and limits (must be defined first for use in later enums)
  enum {
    MAX_UNIT_COUNT        = 128,        // Maximum number of units in allocation
    LARGE_BLOCK_THRESHOLD = 128 * 1024, // 128 KB threshold for large blocks
    MAX_UNITS2INDX_SIZE   = 128         // Size of Units2Indx lookup table
  };

  // Memory allocation unit sizes
  // UNIT_SIZE = 12 bytes (base allocation unit)
  // N_INDEXES = number of free list buckets for different sizes
  enum {
    UNIT_SIZE = 12,
    N1        = 4,
    N2        = 4,
    N3        = 4,
    N4        = (MAX_UNIT_COUNT + 3 - 1 * N1 - 2 * N2 - 3 * N3) / 4,
    N_INDEXES = N1 + N2 + N3 + N4
  };

  // Context flags (PPM_CONTEXT::Flags field)
  enum ContextFlags {
    FLAG_RESCALED      = 0x04, // Context was rescaled
    FLAG_HAS_UPPERCASE = 0x08, // Context contains uppercase symbols
    FLAG_BINARY        = 0x10  // Binary context (2 symbols only)
  };

  // Symbol classification
  enum {
    UPPERCASE_THRESHOLD =
        0x40 // ASCII '@' - symbols >= this are uppercase letters
  };

  // Adaptive learning thresholds for SEE2 context
  enum SEE2Thresholds {
    SEE2_THRESHOLD_LOW  = 40,  // First adaptation threshold
    SEE2_THRESHOLD_MID  = 280, // Second adaptation threshold
    SEE2_THRESHOLD_HIGH = 1020 // Third adaptation threshold
  };

  // Binary model initialization
  enum {
    BIN_SCALE_FACTOR = 128,     // Scaling factor for binary summaries
    BIN_SCALE_MIN    = 32,      // Minimum value for CLAMP
    BIN_SCALE_MAX    = 256 - 32 // Maximum value for CLAMP (224)
  };

  byte *HeapStart; // Start of allocated heap memory

  // Pointer compression: Convert pointers to 32-bit indices to save memory
  // On 64-bit systems, this saves 4 bytes per pointer (50% reduction)
  uint Ptr2Indx(void *p) {
    assert(HeapStart != nullptr && "Heap not initialized in Ptr2Indx");
    assert(UnitsStartBase != nullptr &&
           "UnitsStartBase not initialized in Ptr2Indx");

    // Validate pointer is within heap
    if (p < (void *)HeapStart || p >= (void *)(HeapStart + SubAllocatorSize)) {
      printf("Ptr2Indx ERROR: pointer out of heap bounds\n");
      printf("  p = %p\n", p);
      printf("  HeapStart = %p\n", HeapStart);
      printf("  HeapEnd = %p\n", HeapStart + SubAllocatorSize);
      printf("  SubAllocatorSize = %llu\n", SubAllocatorSize);
      assert(false && "Ptr2Indx: pointer out of heap bounds");
    }

    qword addr = ((byte *)p) - HeapStart;
    // Use fixed baseline so index encoding does not depend on current
    // UnitsStart
    uint lim  = (uint)(UnitsStartBase - HeapStart);
    uint indx = (addr >= lim) ? (addr - lim) / UNIT_SIZE + lim : addr;

    // Validate resulting index would convert back to a valid pointer
    // This catches cases where the index formula would produce out-of-bounds
    // addresses
    qword testAddr = (indx >= lim) ? qword(indx - lim) * UNIT_SIZE + lim : indx;
    if (testAddr >= SubAllocatorSize) {
      printf(
          "Ptr2Indx ERROR: computed index would convert to invalid address\n");
      printf("  p = %p\n", p);
      printf("  addr (p offset) = %llu\n", addr);
      printf("  lim = %u\n", lim);
      printf("  indx (computed) = %u (0x%x)\n", indx, indx);
      printf("  testAddr (convert back) = %llu\n", testAddr);
      printf("  SubAllocatorSize = %llu\n", SubAllocatorSize);

      assert(false && "Ptr2Indx: index would produce out-of-bounds address");
    }

    return indx;
  }

  void *Indx2Ptr(uint indx) {
    // Validate that heap is initialized
    assert(HeapStart != nullptr && "Heap not initialized");
    assert(UnitsStartBase != nullptr && "UnitsStartBase not initialized");

    // Special case: index 0 typically means NULL pointer
    if (indx == 0) {
      return nullptr;
    }

    // Use fixed baseline so decoding is invariant to UnitsStart shifts
    uint  lim  = (uint)(UnitsStartBase - HeapStart);
    qword addr = (indx >= lim) ? qword(indx - lim) * UNIT_SIZE + lim : indx;

    // Validate that computed address is within heap bounds
    // If this triggers, an invalid index was dereferenced.
    if (addr >= SubAllocatorSize) {
      printf("\n*** Indx2Ptr: Invalid index detected ***\n");
      printf("  indx = %u (0x%x)\n", indx, indx);
      printf("  lim = %u (text area size)\n", lim);
      printf("  SubAllocatorSize = %llu\n", SubAllocatorSize);
      printf("  Computed addr = %llu (exceeds heap)\n", addr);

      assert(false && "Indx2Ptr: computed address out of heap bounds");
      return nullptr;
    }

    byte *result = HeapStart + addr;

    // Validate result is within valid memory region
    assert(result >= HeapStart && "Result pointer before heap start");
    assert(result < HeapStart + SubAllocatorSize &&
           "Result pointer beyond heap end");

    return result;
  }

  // Debug helper: validate that an index maps to an address within heap.
  inline void assert_index_valid(uint indx, const char *where) {
    if (indx == 0)
      return;
    // Validate using fixed baseline for stability across UnitsStart shifts
    uint  lim  = (uint)(UnitsStartBase - HeapStart);
    qword addr = (indx >= lim) ? qword(indx - lim) * UNIT_SIZE + lim : indx;
    if (addr >= SubAllocatorSize) {
      printf("\n*** assert_index_valid FAILED at %s ***\n", where);
      printf("  indx = %u (0x%x)\n", indx, indx);
      printf("  lim = %u (text area size)\n", lim);
      printf("  SubAllocatorSize = %llu\n", SubAllocatorSize);
      printf("  Computed addr = %llu (exceeds heap)\n", addr);

      assert(false && "Index maps beyond heap bounds");
    }
  }

  struct _MEM_BLK {
    uint Stamp;
    uint NextIndx;
    uint NU;
  };

  struct BLK_NODE {
    uint Stamp;
    uint NextIndx;
    int  avail() const { return (NextIndx != 0); }
  };

  // DEBUG: Validate BList integrity (called after operations that modify BList)
  void ValidateBList(const char *caller) {
    for (int i = 0; i <= N_INDEXES; i++) {
      uint indx = BList[i].NextIndx;
      if (indx != 0 && (indx == 0x13610002 || indx > SubAllocatorSize)) {
        printf("[ValidateBList] CORRUPTION in BList[%d] after %s\n", i, caller);
        printf("  BList[%d].NextIndx = 0x%x\n", i, indx);
        printf("  BList[%d].Stamp = 0x%x\n", i, BList[i].Stamp);
        assert(false && "BList corrupted");
      }
    }
  }

  BLK_NODE *getNext(BLK_NODE *This) {
    assert(This != nullptr && "getNext: This is null");
    // Note: This can be &BList[i] (struct member), NOT a heap pointer

    // FIX: Sanitize corrupted NextIndx from legacy blocks before conversion
    uint indx = This->NextIndx;
    if (indx != 0) {
      uint  lim  = (uint)(UnitsStartBase - HeapStart);
      qword addr = (indx >= lim) ? qword(indx - lim) * UNIT_SIZE + lim : indx;
      if (addr >= SubAllocatorSize) {
        // Corrupted - would result in out-of-bounds address
        // Return nullptr to indicate end of list
        return nullptr;
      }
    }

    BLK_NODE *result = (BLK_NODE *)Indx2Ptr(indx);
    if (result != nullptr) {
      assert((byte *)result >= HeapStart &&
             (byte *)result < HeapStart + SubAllocatorSize &&
             "getNext: result out of heap bounds");
    }
    return result;
  }

  void setNext(BLK_NODE *This, BLK_NODE *p) {
    assert(This != nullptr && "setNext: This is null");
    // Note: This can be &BList[i] (struct member), NOT a heap pointer
    assert(p != nullptr && "setNext: p is null");
    assert((byte *)p >= HeapStart && (byte *)p < HeapStart + SubAllocatorSize &&
           "setNext: p out of heap bounds");

    uint newIndx = Ptr2Indx(p);

    This->NextIndx = newIndx;
  }

  void link(BLK_NODE *This, BLK_NODE *p) {
    assert(This != nullptr && "link: This is null");
    assert(p != nullptr && "link: p is null");
    // Note: This can be &BList[i] (struct member), NOT a heap pointer
    assert((byte *)p >= HeapStart && (byte *)p < HeapStart + SubAllocatorSize &&
           "link: p out of heap bounds");

    // DEBUG: Validate BEFORE copying to prevent propagating corruption
    if (This->NextIndx > SubAllocatorSize) {
      printf("[link] ERROR: Attempting to copy corrupted NextIndx!\n");
      printf("  This=%p, This->NextIndx=0x%x\n", This, This->NextIndx);
      printf("  p=%p\n", p);
      assert(false && "link: source has corrupted NextIndx");
    }

    p->NextIndx = This->NextIndx;

    // DEBUG: Validate the heap node after write
    if (p->NextIndx > SubAllocatorSize) {
      printf("[link] ERROR: p->NextIndx corrupted after write!\n");
      printf("  Expected: 0x%x, Got: 0x%x\n", This->NextIndx, p->NextIndx);
      assert(false && "link: NextIndx corrupted during write");
    }

    setNext(This, p);
  }

  void unlink(BLK_NODE *This) {
    assert(This != nullptr && "unlink: This is null");
    // Note: This can be &BList[i] (struct member) or a heap pointer
    assert(This->avail() && "unlink: list is empty");
    BLK_NODE *next = getNext(This);
    assert(next != nullptr && "unlink: next is null");

    // FIX: Sanitize corrupted NextIndx from legacy blocks in free list
    //  Check if the computed address would be valid using same logic as
    //  Indx2Ptr
    uint nextIndx = next->NextIndx;
    if (nextIndx != 0) {
      uint  lim  = (uint)(UnitsStartBase - HeapStart);
      qword addr = (nextIndx >= lim) ? qword(nextIndx - lim) * UNIT_SIZE + lim
                                     : nextIndx;
      if (addr >= SubAllocatorSize) {
        // Corrupted - would result in out-of-bounds address
        nextIndx = 0;
      }
    }

    This->NextIndx = nextIndx;
  }

  void *remove(BLK_NODE *This) {
    assert(This != nullptr && "remove: This is null");
    // Note: This can be &BList[i] (struct member), NOT a heap pointer
    assert(This->avail() && "remove: list is empty (no elements to remove)");

    BLK_NODE *p = getNext(This);
    assert(p != nullptr && "remove: getNext returned null");
    assert((byte *)p >= HeapStart && (byte *)p < HeapStart + SubAllocatorSize &&
           "remove: p out of heap bounds");

    unlink(This);
    This->Stamp--;

    return p;
  }

  void insert(BLK_NODE *This, void *pv, int NU) {
    assert(This != nullptr && "insert: This is null");
    assert(pv != nullptr && "insert: pv is null");
    // Note: This is typically &BList[i] (struct member), NOT a heap pointer
    assert((byte *)pv >= HeapStart &&
           (byte *)pv < HeapStart + SubAllocatorSize &&
           "insert: pv out of heap bounds");
    assert(NU > 0 && NU <= MAX_UNIT_COUNT && "insert: invalid NU");

    BLK_NODE *p = (BLK_NODE *)pv;
    link(This, p);
    p->Stamp            = ~uint(0);
    ((_MEM_BLK &)*p).NU = NU;
    This->Stamp++;
  }

  struct MEM_BLK : public BLK_NODE {
    uint NU;
  };

  typedef MEM_BLK *pMEM_BLK;

  BLK_NODE         BList[N_INDEXES + 1];

  uint             GlueCount;
  uint             GlueCount1;
  qword            SubAllocatorSize;
  byte            *pText;
  byte            *UnitsStart;
  // Fixed baseline for index encoding/decoding so stored indices remain
  // valid even if UnitsStart shifts during allocator operations.
  byte *UnitsStartBase;
  byte *LoUnit;
  byte *HiUnit;
  byte *AuxUnit;

  uint  U2B(uint NU) { return 8 * NU + 4 * NU; } // Units to Bytes: NU * 12

  // ⚠️ CRITICAL: Memory allocation function
  // SASize is in MEGABYTES and gets left-shifted by 20 bits (multiplied by
  // 1,048,576) Example: SASize=1024 → 1024 << 20 = 1,073,741,824 bytes = 1 GB
  // Bug history: Was set to 14000 (13.67 GB!), fixed to 1024 (1 GB)
  int StartSubAllocator(qword SASize) {
    qword t   = SASize << 20U; // Convert MB to bytes: SASize * 1,048,576
    HeapStart = new byte[t];   // Allocate main heap

    if (HeapStart == NULL)
      return 0;

    // CRITICAL FIX: Zero the entire heap to prevent stale data corruption
    // Without this, uninitialized memory can contain garbage that gets
    // misinterpreted as valid indices/pointers when blocks are reused
    memset(HeapStart, 0, t);

    SubAllocatorSize = t;
    return 1;
  }

  // Initialize memory layout:
  // HeapStart -> pText (grows up) -> Free Lists -> UnitsStart <- LoUnit/HiUnit
  // <- (grows down)
  void InitSubAllocator() {
    memset(BList, 0, sizeof(BList)); // Clear free list buckets
    HiUnit     = (pText = HeapStart) + SubAllocatorSize;
    qword Diff = SubAllocatorSize / 8 / UNIT_SIZE * 7 *
                 UNIT_SIZE; // 7/8 of heap for contexts
    LoUnit = UnitsStart = HiUnit - Diff;
    // Capture the initial UnitsStart as a fixed baseline for pointer/index
    // conversions so indices remain stable even if UnitsStart moves.
    UnitsStartBase = UnitsStart;
    GlueCount = GlueCount1 = 0;
  }

  qword GetUsedMemory() {
    int   i;
    qword RetVal = SubAllocatorSize - (HiUnit - LoUnit) - (UnitsStart - pText);
    for (i = 0; i < N_INDEXES; i++) {

      RetVal -= qword(Indx2Units[i] * BList[i].Stamp) * 12;
    }
    return RetVal;
  }

  // Free all allocated memory
  void StopSubAllocator() {
    if (SubAllocatorSize)
      SubAllocatorSize = 0, delete[] HeapStart;
  }

  // Defragmentation: Merge adjacent free blocks to reduce fragmentation
  // Called when allocation fails or GlueCount reaches threshold
  void GlueFreeBlocks() {
    uint     i, k, sz;
    MEM_BLK  s0;
    pMEM_BLK p, p0 = &s0, p1;

    assert(LoUnit >= UnitsStart && "LoUnit below UnitsStart in GlueFreeBlocks");
    assert(HiUnit <= HeapStart + SubAllocatorSize &&
           "HiUnit beyond heap in GlueFreeBlocks");

    if (LoUnit != HiUnit)
      LoUnit[0] = 0;

    for (p0->NextIndx = 0, i = 0; i <= N_INDEXES; i++) {
      while (BList[i].avail()) {
        p = (MEM_BLK *)remove(&BList[i]);
        assert(p >= (MEM_BLK *)HeapStart &&
               p < (MEM_BLK *)(HeapStart + SubAllocatorSize) &&
               "GlueFreeBlocks: removed block out of bounds");

        if (p->NU) {
          assert(p->NU > 0 && p->NU <= MAX_UNIT_COUNT &&
                 "Invalid NU in GlueFreeBlocks");

          while (p1 = p + p->NU, p1->Stamp == ~uint(0)) {
            assert((byte *)p1 < HeapStart + SubAllocatorSize &&
                   "Adjacent block beyond heap");
            // Additional validation to avoid false-positive merges when payload
            // happens to start with 0xFFFFFFFF by coincidence.
            if (p1->NU == 0 || p1->NU > MAX_UNIT_COUNT) {
              // Not a sane free block header; stop merging chain here.
              break;
            }
            p->NU += p1->NU;
            p1->NU = 0;
          }
          link(p0, p);
          p0 = p;
        }
      }
    }

    while (s0.avail()) {
      p = (MEM_BLK *)remove(&s0);
      assert(p >= (MEM_BLK *)HeapStart &&
             p < (MEM_BLK *)(HeapStart + SubAllocatorSize) &&
             "GlueFreeBlocks: removed block from s0 out of bounds");

      sz = p->NU;
      if (sz) {
        assert(sz <= MAX_UNIT_COUNT * 100 &&
               "GlueFreeBlocks: sz unreasonably large");

        for (; sz > MAX_UNIT_COUNT; sz -= MAX_UNIT_COUNT, p += MAX_UNIT_COUNT) {
          assert((byte *)p >= HeapStart &&
                 (byte *)p < HeapStart + SubAllocatorSize &&
                 "GlueFreeBlocks: p out of bounds in split loop");
          p->NextIndx = 0; // Clear stale data
          insert(&BList[N_INDEXES - 1], p, MAX_UNIT_COUNT);
        }

        assert(sz > 0 && sz <= MAX_UNIT_COUNT &&
               "GlueFreeBlocks: remaining sz invalid");
        i = Units2Indx[sz - 1];
        assert(i < N_INDEXES && "GlueFreeBlocks: index i out of range");

        if (Indx2Units[i] != sz) {
          assert(i > 0 && "GlueFreeBlocks: cannot decrement i");
          k = sz - Indx2Units[--i];
          assert(k > 0 && k < sz && "GlueFreeBlocks: invalid k calculation");
          assert(k - 1 < N_INDEXES && "GlueFreeBlocks: k-1 out of range");

          MEM_BLK *split_block = p + (sz - k);
          assert((byte *)split_block >= HeapStart &&
                 (byte *)split_block < HeapStart + SubAllocatorSize &&
                 "GlueFreeBlocks: split_block out of bounds");
          split_block->NextIndx = 0; // Clear stale data
          insert(&BList[k - 1], split_block, k);
        }

        assert((byte *)p >= HeapStart &&
               (byte *)p < HeapStart + SubAllocatorSize &&
               "GlueFreeBlocks: p out of bounds before final insert");
        p->NextIndx = 0; // Clear stale data
        insert(&BList[i], p, Indx2Units[i]);
      }
    }

    GlueCount = 1 << (13 + GlueCount1++);
    assert(GlueCount > 0 && "GlueFreeBlocks: GlueCount overflow");
  }

  void SplitBlock(void *pv, uint OldIndx, uint NewIndx) {
    assert(pv >= (void *)HeapStart &&
           pv < (void *)(HeapStart + SubAllocatorSize) &&
           "SplitBlock: pv out of heap bounds");
    assert(OldIndx < N_INDEXES && "SplitBlock: OldIndx out of range");
    assert(NewIndx < N_INDEXES && "SplitBlock: NewIndx out of range");
    assert(OldIndx > NewIndx &&
           "SplitBlock: OldIndx must be greater than NewIndx");

    uint i, k, UDiff = Indx2Units[OldIndx] - Indx2Units[NewIndx];
    assert(UDiff > 0 && UDiff <= MAX_UNIT_COUNT && "SplitBlock: invalid UDiff");

    byte *p = ((byte *)pv) + U2B(Indx2Units[NewIndx]);
    assert(p >= HeapStart && p < HeapStart + SubAllocatorSize &&
           "SplitBlock: computed pointer out of bounds");

    i = Units2Indx[UDiff - 1];
    if (Indx2Units[i] != UDiff) {
      k = Indx2Units[--i];
      assert(k > 0 && k <= MAX_UNIT_COUNT && "SplitBlock: invalid k");
      ((BLK_NODE *)p)->NextIndx = 0; // Clear stale data
      insert(&BList[i], p, k);
      p += U2B(k);
      assert(p >= HeapStart && p < HeapStart + SubAllocatorSize &&
             "SplitBlock: pointer after split out of bounds");
      UDiff -= k;
    }
    assert(UDiff > 0 && "SplitBlock: UDiff became zero");
    ((BLK_NODE *)p)->NextIndx = 0; // Clear stale data
    insert(&BList[Units2Indx[UDiff - 1]], p, UDiff);
  }

  void *AllocUnitsRare(uint indx) {
    assert(indx < N_INDEXES && "Invalid index in AllocUnitsRare");
    assert(HeapStart != nullptr && "Heap not initialized");
    assert(UnitsStart != nullptr && "UnitsStart not initialized");

    uint i = indx;
    do {
      if (++i == N_INDEXES) {
        if (!GlueCount--) {
          GlueFreeBlocks();
          if (BList[i = indx].avail()) {
            void *result = remove(&BList[i]);
            assert(result >= (void *)HeapStart &&
                   result < (void *)(HeapStart + SubAllocatorSize) &&
                   "AllocUnitsRare: result pointer out of heap bounds after "
                   "GlueFreeBlocks");
            return result;
          }
        } else {
          i = U2B(Indx2Units[indx]);
          if (UnitsStart - pText > i) {
            UnitsStart -= i;
            assert(UnitsStart >= pText && "UnitsStart moved below pText");
            assert(UnitsStart < HeapStart + SubAllocatorSize &&
                   "UnitsStart beyond heap");
            return UnitsStart;
          }
          return NULL;
        }
      }
    } while (!BList[i].avail());

    void *RetVal = remove(&BList[i]);
    assert(RetVal >= (void *)HeapStart &&
           RetVal < (void *)(HeapStart + SubAllocatorSize) &&
           "AllocUnitsRare: result pointer out of heap bounds");
    SplitBlock(RetVal, i, indx);

    return RetVal;
  }

  // Allocate NU memory units (NU * 12 bytes)
  // Try free list first, then bump allocate, finally rare path (defragment)
  void *AllocUnits(uint NU) {
    assert(NU > 0 && NU <= MAX_UNIT_COUNT && "Invalid NU in AllocUnits");
    uint indx = Units2Indx[NU - 1];
    assert(indx < N_INDEXES && "Invalid index from Units2Indx");

    if (BList[indx].avail()) {
      void *result = remove(&BList[indx]); // Fast path: reuse free block
      assert(result >= (void *)HeapStart &&
             result < (void *)(HeapStart + SubAllocatorSize) &&
             "AllocUnits: fast path result out of bounds");
      return result;
    }

    void *RetVal = LoUnit;
    LoUnit += U2B(Indx2Units[indx]);
    if (LoUnit <= HiUnit) {
      assert(RetVal >= (void *)UnitsStart && RetVal < (void *)HiUnit &&
             "AllocUnits: bump allocated pointer out of bounds");
      return RetVal; // Bump allocate from bottom
    }

    LoUnit -= U2B(Indx2Units[indx]);
    return AllocUnitsRare(indx); // Slow path: defragment and retry
  }

  // Allocate a PPM_CONTEXT node (grows down from HiUnit)
  void *AllocContext() {
    if (HiUnit != LoUnit) {
      HiUnit -= UNIT_SIZE; // Fast path: bump allocate from top
      assert(HiUnit >= LoUnit && "HiUnit moved below LoUnit");
      assert(HiUnit >= UnitsStart && "HiUnit below UnitsStart");
      return HiUnit;
    }

    void *result =
        BList->avail() ? remove(BList) : AllocUnitsRare(0); // Slow path
    if (result) {
      assert(result >= (void *)HeapStart &&
             result < (void *)(HeapStart + SubAllocatorSize) &&
             "AllocContext: result pointer out of heap bounds");
    }
    return result;
  }

  void FreeUnits(void *ptr, uint NU) {
    assert(ptr >= (void *)HeapStart &&
           ptr < (void *)(HeapStart + SubAllocatorSize) &&
           "FreeUnits: ptr out of heap bounds");
    assert(NU > 0 && NU <= MAX_UNIT_COUNT && "FreeUnits: invalid NU");

    // CRITICAL FIX: Zero the entire block to prevent stale indices from being
    // read later This prevents old iSuccessor/iStats/iSuffix values from
    // causing crashes
    uint blockSize = U2B(NU);
    memset(ptr, 0, blockSize);

    uint indx = Units2Indx[NU - 1];
    assert(indx < N_INDEXES && "FreeUnits: indx out of range");

    insert(&BList[indx], ptr, Indx2Units[indx]);
  }

  void FreeUnit(void *ptr) {
    assert(ptr >= (void *)HeapStart &&
           ptr < (void *)(HeapStart + SubAllocatorSize) &&
           "FreeUnit: ptr out of heap bounds");

    // CRITICAL FIX: Zero the entire block to prevent stale indices
    memset(ptr, 0, UNIT_SIZE);

    int i = (byte *)ptr > UnitsStart + LARGE_BLOCK_THRESHOLD ? 0 : N_INDEXES;
    assert(i >= 0 && i <= N_INDEXES && "FreeUnit: index i out of range");

    insert(&BList[i], ptr, 1);
  }

  void UnitsCpy(void *Dest, void *Src, uint NU) {
    // DEBUG: Check for overlap that could cause corruption
    byte *dst  = (byte *)Dest;
    byte *src  = (byte *)Src;
    uint  size = 12 * NU;

    if (dst < src + size && src < dst + size && dst != src) {
      printf("[UnitsCpy] WARNING: Overlapping copy detected!\\n");
      printf("  Dest=%p, Src=%p, size=%u bytes\\n", Dest, Src, size);
    }

    memcpy(Dest, Src, size);
  }

  void *ExpandUnits(void *OldPtr, uint OldNU) {
    assert(OldPtr >= (void *)HeapStart &&
           OldPtr < (void *)(HeapStart + SubAllocatorSize) &&
           "ExpandUnits: OldPtr out of heap bounds");
    assert(OldNU > 0 && OldNU < MAX_UNIT_COUNT && "ExpandUnits: invalid OldNU");

    uint i0 = Units2Indx[OldNU - 1];
    uint i1 = Units2Indx[OldNU - 1 + 1];
    assert(i0 < N_INDEXES && "ExpandUnits: i0 out of range");
    assert(i1 < N_INDEXES && "ExpandUnits: i1 out of range");

    if (i0 == i1)
      return OldPtr;
    void *ptr = AllocUnits(OldNU + 1);
    if (ptr) {
      UnitsCpy(ptr, OldPtr, OldNU);
      ((BLK_NODE *)OldPtr)->NextIndx = 0; // Clear stale data
      insert(&BList[i0], OldPtr, OldNU);
    }
    return ptr;
  }

  void *ShrinkUnits(void *OldPtr, uint OldNU, uint NewNU) {
    assert(OldPtr >= (void *)HeapStart &&
           OldPtr < (void *)(HeapStart + SubAllocatorSize) &&
           "ShrinkUnits: OldPtr out of heap bounds");
    assert(OldNU > 0 && OldNU <= MAX_UNIT_COUNT &&
           "ShrinkUnits: invalid OldNU");
    assert(NewNU > 0 && NewNU <= MAX_UNIT_COUNT &&
           "ShrinkUnits: invalid NewNU");
    // Allow equal sizes (no-op path handled below when i0 == i1)
    assert(NewNU <= OldNU && "ShrinkUnits: NewNU must be <= OldNU");

    uint i0 = Units2Indx[OldNU - 1];
    uint i1 = Units2Indx[NewNU - 1];
    assert(i0 < N_INDEXES && "ShrinkUnits: i0 out of range");
    assert(i1 < N_INDEXES && "ShrinkUnits: i1 out of range");

    if (i0 == i1)
      return OldPtr;
    if (BList[i1].avail()) {
      void *ptr = remove(&BList[i1]);
      UnitsCpy(ptr, OldPtr, NewNU);
      ((BLK_NODE *)OldPtr)->NextIndx = 0; // Clear stale data
      insert(&BList[i0], OldPtr, Indx2Units[i0]);
      return ptr;
    } else {
      SplitBlock(OldPtr, i0, i1);
      return OldPtr;
    }
  }

  void *MoveUnitsUp(void *OldPtr, uint NU) {
    assert(OldPtr >= (void *)HeapStart &&
           OldPtr < (void *)(HeapStart + SubAllocatorSize) &&
           "MoveUnitsUp: OldPtr out of heap bounds");
    assert(NU > 0 && NU <= MAX_UNIT_COUNT && "Invalid NU in MoveUnitsUp");

    uint indx = Units2Indx[NU - 1];
    assert(indx < N_INDEXES && "MoveUnitsUp: indx out of range");

    PrefetchData(OldPtr);
    if ((byte *)OldPtr > UnitsStart + LARGE_BLOCK_THRESHOLD ||
        (BLK_NODE *)OldPtr > getNext(&BList[indx]))
      return OldPtr;

    void *ptr = remove(&BList[indx]);
    assert(ptr >= (void *)HeapStart &&
           ptr < (void *)(HeapStart + SubAllocatorSize) &&
           "MoveUnitsUp: new pointer out of heap bounds");

    UnitsCpy(ptr, OldPtr, NU);

    ((BLK_NODE *)OldPtr)->NextIndx = 0; // Clear stale data
    insert(&BList[N_INDEXES], OldPtr, Indx2Units[indx]);

    return ptr;
  }

  void PrepareTextArea() {
    assert(UnitsStart >= pText && "UnitsStart below pText in PrepareTextArea");
    assert(UnitsStart < HeapStart + SubAllocatorSize &&
           "UnitsStart beyond heap in PrepareTextArea");

    AuxUnit = (byte *)AllocContext();
    if (!AuxUnit) {
      AuxUnit = UnitsStart;
      assert(AuxUnit >= pText && "AuxUnit (=UnitsStart) below pText");
    } else {
      assert(AuxUnit >= HeapStart && AuxUnit < HeapStart + SubAllocatorSize &&
             "AuxUnit from AllocContext out of bounds");

      if (AuxUnit == UnitsStart) {
        UnitsStart += UNIT_SIZE;
        assert(UnitsStart < HeapStart + SubAllocatorSize &&
               "UnitsStart beyond heap after adjustment");
        AuxUnit = UnitsStart;
      }
    }

    assert(AuxUnit >= pText && AuxUnit < HeapStart + SubAllocatorSize &&
           "AuxUnit final value out of bounds");
  }

  void ExpandTextArea() {
    BLK_NODE *p;
    uint      Count[N_INDEXES], i = 0;
    memset(Count, 0, sizeof(Count));

    assert(UnitsStart >= pText && "UnitsStart below pText before expansion");
    assert(UnitsStart < HeapStart + SubAllocatorSize &&
           "UnitsStart beyond heap before expansion");

    if (AuxUnit != UnitsStart) {
      if (*(uint *)AuxUnit != ~uint(0)) {
        UnitsStart += UNIT_SIZE;
        assert(UnitsStart < HeapStart + SubAllocatorSize &&
               "UnitsStart beyond heap after adjustment");
      } else {
        assert(AuxUnit >= pText && AuxUnit < HeapStart + SubAllocatorSize &&
               "ExpandTextArea: AuxUnit out of heap bounds");
        insert(BList, AuxUnit, 1);
      }
    }

    while ((p = (BLK_NODE *)UnitsStart)->Stamp == ~uint(0)) {
      MEM_BLK *pm = (MEM_BLK *)p;
      assert(pm->NU > 0 && pm->NU <= MAX_UNIT_COUNT &&
             "Invalid NU in ExpandTextArea");

      byte *newUnitsStart = (byte *)(pm + pm->NU);
      assert(newUnitsStart >= pText && "New UnitsStart below pText");
      assert(newUnitsStart < HeapStart + SubAllocatorSize &&
             "New UnitsStart beyond heap");

      UnitsStart = newUnitsStart;
      Count[Units2Indx[pm->NU - 1]]++;
      i++;
      pm->Stamp = 0;
    }

    if (i) {

      for (p = BList + N_INDEXES; p->NextIndx; p = getNext(p)) {
        while (p->NextIndx && !getNext(p)->Stamp) {
          Count[Units2Indx[((MEM_BLK *)getNext(p))->NU - 1]]--;
          unlink(p);
          BList[N_INDEXES].Stamp--;
        }
        if (!p->NextIndx)
          break;
      }

      for (i = 0; i < N_INDEXES; i++) {
        for (p = BList + i; Count[i] != 0; p = getNext(p)) {
          while (!getNext(p)->Stamp) {
            unlink(p);
            BList[i].Stamp--;
            if (!--Count[i])
              break;
          }
        }
      }
    }
  }

  static const int     MAX_O = ORealMAX;

  template <class T> T CLAMP(const T &X, const T &LoX, const T &HiX) {
    return (X >= LoX) ? ((X <= HiX) ? (X) : (HiX)) : (LoX);
  }

  template <class T> void SWAP(T &t1, T &t2) {
    T tmp = t1;
    t1    = t2;
    t2    = tmp;
  }

  void PrefetchData(void *Addr) { *(volatile byte *)Addr; }

  enum { UP_FREQ = 5 };

  byte Indx2Units[N_INDEXES];
  byte Units2Indx[MAX_UNITS2INDX_SIZE];

  byte NS2BSIndx[256];
  byte QTable[260];

  void PPMD_STARTUP(void) {
    int i, k, m, Step;

    for (i = 0, k = 1; i < N1; i++, k += 1)
      Indx2Units[i] = k;
    for (k++; i < N1 + N2; i++, k += 2)
      Indx2Units[i] = k;
    for (k++; i < N1 + N2 + N3; i++, k += 3)
      Indx2Units[i] = k;
    for (k++; i < N1 + N2 + N3 + N4; i++, k += 4)
      Indx2Units[i] = k;

    for (k = 0, i = 0; k < MAX_UNITS2INDX_SIZE; k++) {
      i += Indx2Units[i] < k + 1;
      Units2Indx[k] = i;
    }

    NS2BSIndx[0] = 2 * 0;
    NS2BSIndx[1] = 2 * 1;
    NS2BSIndx[2] = 2 * 1;
    memset(NS2BSIndx + 3, 2 * 2, 26);
    memset(NS2BSIndx + 29, 2 * 3, 256 - 29);

    for (i = 0; i < UP_FREQ; i++)
      QTable[i] = i;

    for (m = i = UP_FREQ, k = Step = 1; i < 260; i++) {
      QTable[i] = m;
      if (!--k)
        k = ++Step, m++;
    }
  }

  enum {
    MAX_FREQ = 124,
    O_BOUND  = 12 // Order Boundary. Low-order contexts < O_BOUND are CRITICAL -
                  // they must not be freed during model restoration
  };

  struct PPM_CONTEXT;

  // STATE: Statistics for one symbol in a context
  // Contains the symbol value, its frequency count, and pointer to next-order
  // context
  struct STATE {
    byte Symbol;     // Byte value (0-255)
    byte Freq;       // Frequency count (how often seen in this context)
    uint iSuccessor; // Index to next-level context (higher order)
  };

  PPM_CONTEXT *getSucc(STATE *This) {
    assert(This != nullptr && "getSucc: This is null");
    if ((byte *)This < HeapStart ||
        (byte *)This >= HeapStart + SubAllocatorSize) {
      printf("\n*** getSucc: Invalid This pointer ***\n");
      printf("  This = %p\n", This);
      printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
             HeapStart + SubAllocatorSize);

      assert(false && "getSucc: This pointer out of heap bounds");
    }
    // Validate index before conversion for clearer diagnostics
    assert_index_valid(This->iSuccessor, "getSucc.iSuccessor");

    PPM_CONTEXT *result = (PPM_CONTEXT *)Indx2Ptr(This->iSuccessor);

    if (result != nullptr) {
      if ((byte *)result < HeapStart ||
          (byte *)result >= HeapStart + SubAllocatorSize) {
        printf("\n*** getSucc: Invalid result pointer ***\n");
        printf("  This = %p\n", This);
        printf("  This->iSuccessor = 0x%x\n", This->iSuccessor);
        printf("  result = %p\n", result);
        printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
               HeapStart + SubAllocatorSize);

        assert(false && "getSucc: result pointer out of heap bounds");
      }
    }

    return result;
  }

  void SWAP(STATE &s1, STATE &s2) {
    word t1       = (word &)s1;
    uint t2       = s1.iSuccessor;
    (word &)s1    = (word &)s2;
    s1.iSuccessor = s2.iSuccessor;
    (word &)s2    = t1;
    s2.iSuccessor = t2;
  }

  // PPM_CONTEXT: A node in the context tree
  // Stores statistics for all symbols seen in this specific context
  // Optimization: When NumStats==0 (binary context), SummFreq stores STATE
  // directly
  struct PPM_CONTEXT {

    byte NumStats; // Number of different symbols (0 = binary optimization)
    byte Flags;    // Status: FLAG_RESCALED, FLAG_HAS_UPPERCASE, FLAG_BINARY
    word SummFreq; // Sum of all symbol frequencies (or STATE if NumStats==0)
    uint iStats;   // Index to STATE array
    uint iSuffix;  // Index to parent context (shorter/lower order)

    // Binary context optimization: reuse SummFreq field to store single STATE
    STATE &oneState() const { return (STATE &)SummFreq; }
  };

  STATE *getStats(PPM_CONTEXT *This) {
    assert(This != nullptr && "getStats: This is null");
    if ((byte *)This < HeapStart ||
        (byte *)This >= HeapStart + SubAllocatorSize) {
      printf("\n*** getStats: Invalid This pointer ***\n");
      printf("  This = %p\n", This);
      printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
             HeapStart + SubAllocatorSize);

      assert(false && "getStats: This pointer out of heap bounds");
    }
    assert_index_valid(This->iStats, "getStats.iStats");

    STATE *result = (STATE *)Indx2Ptr(This->iStats);

    if (result != nullptr && This->NumStats > 0) {
      if ((byte *)result < HeapStart ||
          (byte *)result >= HeapStart + SubAllocatorSize) {
        printf("\n*** getStats: Invalid result pointer ***\n");
        printf("  This = %p\n", This);
        printf("  This->iStats = 0x%x\n", This->iStats);
        printf("  This->NumStats = %d\n", This->NumStats);
        printf("  result = %p\n", result);
        printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
               HeapStart + SubAllocatorSize);

        assert(false && "getStats: result pointer out of heap bounds");
      }
    }

    return result;
  }

  PPM_CONTEXT *suff(PPM_CONTEXT *This) {
    assert(This != nullptr && "suff: This is null");
    if ((byte *)This < HeapStart ||
        (byte *)This >= HeapStart + SubAllocatorSize) {
      printf("\n*** suff: Invalid This pointer ***\n");
      printf("  This = %p\n", This);
      printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
             HeapStart + SubAllocatorSize);

      assert(false && "suff: This pointer out of heap bounds");
    }
    // printf("suff: This=%p NumStats=%u iSuffix=0x%x\n", This,
    // (unsigned)This->NumStats, This->iSuffix);
    assert_index_valid(This->iSuffix, "suff.iSuffix");

    PPM_CONTEXT *result = (PPM_CONTEXT *)Indx2Ptr(This->iSuffix);

    if (result != nullptr) {
      if ((byte *)result < HeapStart ||
          (byte *)result >= HeapStart + SubAllocatorSize) {
        printf("\n*** suff: Invalid result pointer ***\n");
        printf("  This = %p\n", This);
        printf("  This->iSuffix = 0x%x\n", This->iSuffix);
        printf("  result = %p\n", result);
        printf("  HeapStart = %p, HeapEnd = %p\n", HeapStart,
               HeapStart + SubAllocatorSize);

        assert(false && "suff: result pointer out of heap bounds");
      }
    }

    return result;
  }

  int          _MaxOrder, _CutOff, _MMAX;
  uint         _filesize;
  int          OrderFall;

  STATE       *FoundState;
  PPM_CONTEXT *MaxContext;

  uint         EscCount;
  uint         CharMask[256];

  int          BSumm;
  int          RunLength;
  int          InitRL;

  enum {
    INT_BITS    = 7,
    PERIOD_BITS = 7,
    TOT_BITS    = INT_BITS + PERIOD_BITS,
    INTERVAL    = 1 << INT_BITS,
    BIN_SCALE   = 1 << TOT_BITS,
    ROUND       = 16
  };

  // SEE2_CONTEXT: Secondary Escape Estimation
  // Used to estimate escape probabilities more accurately
  // Adapts based on context usage patterns
  struct SEE2_CONTEXT {
    word Summ;  // Sum of escape frequencies
    byte Shift; // Scaling shift value (adaptive)
    byte Count; // Update counter

    void init(uint InitVal) {
      Shift = PERIOD_BITS - 4;
      Summ  = InitVal << Shift;
      Count = 7;
    }

    uint getMean() { return Summ >> Shift; } // Get average escape probability

    void update() {
      if (--Count == 0)
        setShift_rare(); // Adjust scaling after period
    }

    void setShift_rare() {
      uint i = Summ >> Shift;
      i = PERIOD_BITS - (i > SEE2_THRESHOLD_LOW) - (i > SEE2_THRESHOLD_MID) -
          (i > SEE2_THRESHOLD_HIGH);
      if (i < Shift) {
        Summ >>= 1;
        Shift--;
      } else if (i > Shift) {
        Summ <<= 1;
        Shift++;
      }
      Count = 5 << Shift;
    }
  };

  int NumMasked; // Number of symbols excluded (already tried in longer
                 // contexts)

  // Frequency rescaling: Called when frequencies get too high
  // Divides all frequencies by 2 to prevent overflow while maintaining ratios
  STATE *rescale(PPM_CONTEXT &q, int OrderFall, STATE *FoundState) {
    STATE  tmp;
    STATE *p;
    STATE *p1;

    q.Flags &=
        (FLAG_BINARY | FLAG_RESCALED); // Keep only binary and rescaled flags

    p1  = getStats(&q);
    tmp = FoundState[0];
    for (p = FoundState; p != p1; p--)
      p[0] = p[-1];
    p1[0]  = tmp;

    int of = (OrderFall != 0);
    int a, i;
    int f0      = p->Freq;
    int sf      = q.SummFreq;
    int EscFreq = sf - f0;
    q.SummFreq = p->Freq = (f0 + of) >> 1;

    for (i = 0; i < q.NumStats; i++) {
      p++;
      a = p->Freq;
      EscFreq -= a;
      a       = (a + of) >> 1;
      p->Freq = a;
      q.SummFreq += a;
      if (a)
        q.Flags |= FLAG_HAS_UPPERCASE * (p->Symbol >= UPPERCASE_THRESHOLD);
      if (a > p[-1].Freq) {
        tmp = p[0];
        for (p1 = p; tmp.Freq > p1[-1].Freq; p1--)
          p1[0] = p1[-1];
        p1[0] = tmp;
      }
    }

    if (p->Freq == 0) {
      for (i = 0; p->Freq == 0; i++, p--)
        ;
      EscFreq += i;
      a = (q.NumStats + 2) >> 1;
      if ((q.NumStats -= i) == 0) {
        tmp      = getStats(&q)[0];
        tmp.Freq = Min(MAX_FREQ / 3, (2 * tmp.Freq + EscFreq - 1) / EscFreq);
        q.Flags &= 0x18;
        FreeUnits(getStats(&q), a);
        q.oneState() = tmp;
        FoundState   = &q.oneState();
        return FoundState;
      }
      q.iStats = Ptr2Indx(ShrinkUnits(getStats(&q), a, (q.NumStats + 2) >> 1));
    }

    q.SummFreq += (EscFreq + 1) >> 1;
    if (OrderFall || (q.Flags & 0x04) == 0) {
      a = (sf -= EscFreq) - f0;
      a = CLAMP(uint((f0 * q.SummFreq - sf * getStats(&q)->Freq + a - 1) / a),
                2U, MAX_FREQ / 2U - 18U);
    } else {
      a = 2;
    }

    (FoundState = getStats(&q))->Freq += a;
    q.SummFreq += a;
    q.Flags |= 0x04;

    return FoundState;
  }

  void AuxCutOff(STATE *p, int Order, int MaxOrder) {
    if (Order < MaxOrder) {
      PPM_CONTEXT *succ = getSucc(p);
      if (succ != nullptr) {
        PrefetchData(succ);
        p->iSuccessor = cutOff(succ[0], Order + 1, MaxOrder);
      } else {
        assert(false && "AuxCutOff: stale/null successor");
      }
    } else {
      p->iSuccessor = 0;
    }
  }

  // Memory cutoff: Remove rarely-used contexts to free memory
  // Called when memory is exhausted to make room for new contexts
  uint cutOff(PPM_CONTEXT &q, int Order, int MaxOrder) {
    int    i, tmp, EscFreq, Scale;
    STATE *p;
    STATE *p0;

    assert(Order >= 0 && Order <= MaxOrder && "cutOff: invalid Order");
    assert(MaxOrder > 0 && MaxOrder <= 255 && "cutOff: invalid MaxOrder");
    assert(&q >= (PPM_CONTEXT *)HeapStart &&
           &q < (PPM_CONTEXT *)(HeapStart + SubAllocatorSize) &&
           "cutOff: context out of heap bounds");

    //

    if (q.NumStats == 0) {
      int flag = 1;
      p        = &q.oneState();
      assert(p >= (STATE *)HeapStart &&
             p < (STATE *)(HeapStart + SubAllocatorSize) &&
             "cutOff: oneState pointer out of bounds");

      PPM_CONTEXT *succ = getSucc(p);
      if (succ != nullptr && (byte *)succ >= UnitsStart) {
        assert(succ >= (PPM_CONTEXT *)HeapStart &&
               succ < (PPM_CONTEXT *)(HeapStart + SubAllocatorSize) &&
               "cutOff: successor context out of bounds");
        AuxCutOff(p, Order, MaxOrder);
        if (p->iSuccessor || Order < O_BOUND)
          flag = 0;
      }
      if (flag) {
        FreeUnit(&q);
        return 0;
      }

    } else {
      assert(q.NumStats > 0 && q.NumStats <= 255 && "cutOff: invalid NumStats");

      tmp = (q.NumStats + 2) >> 1;
      assert(tmp > 0 && tmp <= 129 && "cutOff: invalid tmp calculation");

      STATE *stats_before = getStats(&q);
      assert(stats_before >= (STATE *)HeapStart &&
             stats_before < (STATE *)(HeapStart + SubAllocatorSize) &&
             "cutOff: getStats result out of bounds before MoveUnitsUp");

      p0 = (STATE *)MoveUnitsUp(getStats(&q), tmp);
      // Validate that p0 points to valid heap memory (it may legitimately be
      // either below or above current HiUnit due to allocator movements)
      assert((byte *)p0 >= HeapStart &&
             (byte *)p0 < HeapStart + SubAllocatorSize &&
             "cutOff: p0 outside heap bounds");

      q.iStats = Ptr2Indx(p0);

      // suspicious code - modifying loop variable inside loop
      // possibility that p is beyond bounds
      for (i = q.NumStats, p = &p0[i]; p >= p0; p--) {
        // Validate pointer is within heap; position relative to
        // UnitsStart/HiUnit can vary
        assert((byte *)p >= HeapStart &&
               (byte *)p < HeapStart + SubAllocatorSize &&
               "cutOff: p outside heap bounds in loop");

        PPM_CONTEXT *succCtx = getSucc(p);
        byte        *succ    = (byte *)succCtx;
        // In pruning: a null successor (index==0) or a text-area successor
        // (< UnitsStart) should be dropped, not asserted.
        if (succCtx == nullptr || succ < UnitsStart) {
          p[0].iSuccessor = 0;
          SWAP(p[0], p0[i--]);
        } else {
          AuxCutOff(p, Order, MaxOrder);
        }
      }

      if (i != q.NumStats && Order > 0) {
        q.NumStats = i;
        p          = p0;
        if (i < 0) {
          FreeUnits(p, tmp);
          FreeUnit(&q);
          return 0;
        }
        if (i == 0) {
          q.Flags = (q.Flags & FLAG_BINARY) +
                    FLAG_HAS_UPPERCASE * (p[0].Symbol >= UPPERCASE_THRESHOLD);
          p[0].Freq    = 1 + (2 * (p[0].Freq - 1)) / (q.SummFreq - p[0].Freq);
          q.oneState() = p[0];
          FreeUnits(p, tmp);
        } else {
          uint new_tmp = (i + 2) >> 1;
          assert(new_tmp > 0 && new_tmp <= tmp &&
                 "cutOff: invalid new_tmp for ShrinkUnits");

          p = (STATE *)ShrinkUnits(p0, tmp, new_tmp);

          assert(p >= (STATE *)HeapStart &&
                 p < (STATE *)(HeapStart + SubAllocatorSize) &&
                 "cutOff: p after ShrinkUnits out of bounds");

          q.iStats = Ptr2Indx(p);
          Scale    = (q.SummFreq > 16 * i);
          q.Flags  = (q.Flags & (FLAG_BINARY + FLAG_RESCALED * Scale));
          if (Scale) {
            EscFreq    = q.SummFreq;
            q.SummFreq = 0;
            for (i = 0; i <= q.NumStats; i++) {
              EscFreq -= p[i].Freq;
              p[i].Freq = (p[i].Freq + 1) >> 1;
              q.SummFreq += p[i].Freq;
              q.Flags |=
                  FLAG_HAS_UPPERCASE * (p[i].Symbol >= UPPERCASE_THRESHOLD);
            };
            EscFreq = (EscFreq + 1) >> 1;
            q.SummFreq += EscFreq;
          } else {
            for (i = 0; i <= q.NumStats; i++)
              q.Flags |=
                  FLAG_HAS_UPPERCASE * (p[i].Symbol >= UPPERCASE_THRESHOLD);
          }
        }
      }
    }

    if ((byte *)&q == UnitsStart) {

      UnitsCpy(AuxUnit, &q, 1);
      return Ptr2Indx(AuxUnit);
    } else {

      if ((byte *)suff(&q) == UnitsStart)
        q.iSuffix = Ptr2Indx(AuxUnit);
    }

    return Ptr2Indx(&q);
  }

  void StartModelRare(void) {
    int  i, k, s;
    byte i2f[25];

    memset(CharMask, 0, sizeof(CharMask));
    EscCount = 1;

    if (_MaxOrder < 2) {
      OrderFall = _MaxOrder;
      for (PPM_CONTEXT *pc = MaxContext; pc->iSuffix != 0; pc = suff(pc))
        OrderFall--;
      return;
    }

    OrderFall = _MaxOrder;

    InitSubAllocator();

    InitRL               = -((_MaxOrder < 13) ? _MaxOrder : 13);
    RunLength            = InitRL;

    MaxContext           = (PPM_CONTEXT *)AllocContext();
    MaxContext->NumStats = 255;
    MaxContext->SummFreq = 255 + 2;
    MaxContext->iStats   = Ptr2Indx(AllocUnits(256 / 2));
    MaxContext->Flags    = 0;
    MaxContext->iSuffix  = 0;
    PrevSuccess          = 0;

    for (i = 0; i < 256; i++) {
      getStats(MaxContext)[i].Symbol     = i;
      getStats(MaxContext)[i].Freq       = 1;
      getStats(MaxContext)[i].iSuccessor = 0;
    }

    if (1) {

      for (k = i = 0; i < 25; i2f[i++] = k + 1)
        while (QTable[k] == i)
          k++;

      for (k = 0; k < 64; k++) {
        for (s = i = 0; i < 6; i++)
          s += EscCoef[2 * i + ((k >> i) & 1)];
        s = BIN_SCALE_FACTOR * CLAMP(s, (int)BIN_SCALE_MIN, (int)BIN_SCALE_MAX);
        for (i = 0; i < 25; i++)
          BinSumm[i][k] = BIN_SCALE - s / i2f[i];
      }

      for (i = 0; i < 23; i++)
        for (k = 0; k < 32; k++)
          SEE2Cont[i][k].init(8 * i + 5);
    }
  }

  void RestoreModelRare(void) {
    STATE *p;
    pText           = HeapStart;
    PPM_CONTEXT *pc = saved_pc;

    for (;; MaxContext = suff(MaxContext)) {
      if ((MaxContext->NumStats == 1) && (MaxContext != pc)) {
        p                     = getStats(MaxContext);
        PPM_CONTEXT *nextSucc = getSucc(p + 1);
        if (nextSucc != nullptr && (byte *)(nextSucc) >= UnitsStart)
          break;
      } else
        break;

      MaxContext->Flags =
          (MaxContext->Flags & FLAG_BINARY) +
          FLAG_HAS_UPPERCASE * (p->Symbol >= UPPERCASE_THRESHOLD);
      p[0].Freq              = (p[0].Freq + 1) >> 1;
      MaxContext->oneState() = p[0];
      MaxContext->NumStats   = 0;
      FreeUnits(p, 1);
    }

    while (MaxContext->iSuffix)
      MaxContext = suff(MaxContext);

    AuxUnit = UnitsStart;

    ExpandTextArea();

    do {
      PrepareTextArea();
      cutOff(MaxContext[0], 0, _MaxOrder);
      ExpandTextArea();
    } while (GetUsedMemory() > 3 * (SubAllocatorSize >> 2));

    GlueCount = GlueCount1 = 0;
    OrderFall              = _MaxOrder;
  }

  PPM_CONTEXT *saved_pc;

  PPM_CONTEXT *UpdateModel(PPM_CONTEXT *MinContext) {
    byte         Flag;        // context flags
    byte         FSymbol;     // found symbol
    uint         ns1;         // number of statistics in the successor context
    uint         ns;          // number of statistics in the current context
    uint         cf;          // comparison factor
    uint         sf;          // scaling factor
    uint         s0;          // sum frequency minus found frequency
    uint         FFreq;       // found frequency
    uint         iSuccessor;  // index of successor context
    uint         iFSuccessor; // index of found successor context
    PPM_CONTEXT *pc = NULL;   // context pointer
    STATE       *p  = NULL;   // state pointer

    FSymbol         = FoundState->Symbol;
    FFreq           = FoundState->Freq;
    iFSuccessor     = FoundState->iSuccessor;

    if (MinContext->iSuffix) {
      pc = suff(MinContext);

      if (pc[0].NumStats) {
        p = getStats(pc);
        if (p[0].Symbol != FSymbol) {
          for (p++; p[0].Symbol != FSymbol; p++)
            ;
          if (p[0].Freq >= p[-1].Freq)
            SWAP(p[0], p[-1]), p--;
        }
        if (p[0].Freq < MAX_FREQ - 3) {
          cf = 2 + (FFreq < 28);
          p[0].Freq += cf;
          pc[0].SummFreq += cf;
        }
      } else {
        p = &(pc[0].oneState());
        p[0].Freq += (p[0].Freq < 14);
      }
    }
    // pc = MaxContext;

    if (!OrderFall && iFSuccessor) {
      assert_index_valid(iFSuccessor, "UpdateModel.iFSuccessor (early)");
      FoundState->iSuccessor = CreateSuccessors(1, p, MinContext);
      if (!FoundState->iSuccessor) {
        saved_pc = pc;
        return 0;
      };
      MaxContext = getSucc(FoundState);
      if (MaxContext == nullptr) {
        return 0;
      }
      return MaxContext;
    }

    *pText++   = FSymbol;
    iSuccessor = Ptr2Indx(pText);
    if (pText >= UnitsStart) {
      saved_pc = pc;
      return 0;
    };

    if (iFSuccessor) {
      assert_index_valid(iFSuccessor, "UpdateModel.iFSuccessor (prefetch)");
      void *iSuccPtr = Indx2Ptr(iFSuccessor);
      if (iSuccPtr == nullptr) {
        // Stale index - recreate successor
        iFSuccessor = CreateSuccessors(0, p, MinContext);
      } else if ((byte *)iSuccPtr < UnitsStart) {
        iFSuccessor = CreateSuccessors(0, p, MinContext);
      } else {
        PrefetchData(iSuccPtr);
      }
    } else {
      iFSuccessor = ReduceOrder(p, MinContext);
    }

    if (!iFSuccessor) {
      saved_pc = pc;
      return 0;
    };

    if (!--OrderFall) {
      iSuccessor = iFSuccessor;
      pText -= (MaxContext != MinContext);
    }

    s0   = MinContext->SummFreq - FFreq;
    ns   = MinContext->NumStats;
    Flag = FLAG_HAS_UPPERCASE * (FSymbol >= UPPERCASE_THRESHOLD);
    for (pc = MaxContext; pc != MinContext; pc = suff(pc)) {
      ns1 = pc[0].NumStats;

      if (ns1) {

        if (ns1 & 1) {
          p = (STATE *)ExpandUnits(getStats(pc), (ns1 + 1) >> 1);
          if (!p) {
            saved_pc = pc;
            return 0;
          };
          pc[0].iStats = Ptr2Indx(p);
        }

        pc[0].SummFreq += QTable[ns + 4] >> 3;
      } else {
        p = (STATE *)AllocUnits(1);
        if (!p) {
          saved_pc = pc;
          return 0;
        };
        p[0]         = pc[0].oneState();
        pc[0].iStats = Ptr2Indx(p);
        p[0].Freq =
            (p[0].Freq <= MAX_FREQ / 3) ? (2 * p[0].Freq - 1) : (MAX_FREQ - 15);

        pc[0].SummFreq = p[0].Freq + (ns > 1) + ExpEscape[QTable[BSumm >> 8]];
      }

      cf = (FFreq - 1) * (5 + pc[0].SummFreq);
      sf = s0 + pc[0].SummFreq;

      if (cf <= 3 * sf) {

        cf = 1 + (2 * cf > sf) + (2 * cf > 3 * sf);
        pc[0].SummFreq += 4;
      } else {
        cf = 5 + (cf > 5 * sf) + (cf > 6 * sf) + (cf > 8 * sf) +
             (cf > 10 * sf) + (cf > 12 * sf);
        pc[0].SummFreq += cf;
      }

      // this is very suspicious
      // after this point, p seems to be invalid when we are near memory limit
      p = getStats(pc) + (++pc[0].NumStats);

      // Sanity check: stats write must remain within overall heap bounds
      if ((byte *)p < HeapStart ||
          (byte *)p + sizeof(STATE) > HeapStart + SubAllocatorSize) {
        assert(false && "UpdateModel: stats pointer out of heap bounds");
        saved_pc = pc;
        return 0;
      }

      p[0].iSuccessor = iSuccessor;
      p[0].Symbol     = FSymbol;
      p[0].Freq       = cf;
      pc[0].Flags |= Flag;
    }

    assert_index_valid(iFSuccessor, "UpdateModel.iFSuccessor (final)");
    MaxContext = (PPM_CONTEXT *)Indx2Ptr(iFSuccessor);
    return MaxContext;
  }

  uint CreateSuccessors(uint Skip, STATE *p, PPM_CONTEXT *pc) {
    byte    tmp;
    uint    cf, s0;
    STATE  *ps[MAX_O];
    STATE **pps       = ps;

    byte    sym       = FoundState->Symbol;
    uint    iUpBranch = FoundState->iSuccessor;
    assert_index_valid(iUpBranch, "CreateSuccessors.iUpBranch");

    if (!Skip) {
      *pps++ = FoundState;
      if (!pc[0].iSuffix)
        goto NO_LOOP;
    }

    if (p) {
      pc = suff(pc);
      goto LOOP_ENTRY;
    }

    do {
      pc = suff(pc);
      if (!pc) {
        assert(false && "CreateSuccessors: null suffix context");
        return 0;
      }

      if (pc[0].NumStats) {

        p = getStats(pc);
        if (!p) {
          assert(false && "CreateSuccessors: null stats pointer");
          return 0;
        }
        for (; p[0].Symbol != sym; p++)
          ;

        tmp = 2 * (p[0].Freq < MAX_FREQ - 1);
        p[0].Freq += tmp;
        pc[0].SummFreq += tmp;
      } else {

        p                    = &(pc[0].oneState());
        PPM_CONTEXT *pc_suff = suff(pc);
        if (!pc_suff) {
          assert(false && "CreateSuccessors: null suffix in binary case");
          return 0;
        }
        p[0].Freq += (!pc_suff->NumStats & (p[0].Freq < 16));
      }

    LOOP_ENTRY:
      if (p[0].iSuccessor != iUpBranch) {
        pc = getSucc(p);
        if (pc == nullptr) {
          assert(false && "CreateSuccessors: null successor context");
          return 0;
        }
        break;
      }
      *pps++ = p;
    } while (pc[0].iSuffix);

  NO_LOOP:
    if (pps == ps)
      return Ptr2Indx(pc);

    PPM_CONTEXT ct;
    ct.NumStats              = 0;
    ct.Flags                 = FLAG_BINARY * (sym >= UPPERCASE_THRESHOLD);
    sym                      = *(byte *)Indx2Ptr(iUpBranch);
    ct.oneState().iSuccessor = Ptr2Indx((byte *)Indx2Ptr(iUpBranch) + 1);
    ct.oneState().Symbol     = sym;
    ct.Flags |= FLAG_HAS_UPPERCASE * (sym >= UPPERCASE_THRESHOLD);

    if (pc[0].NumStats) {
      p = getStats(pc);
      if (!p) {
        assert(false && "CreateSuccessors: null stats pointer at suffix step");
        return 0;
      }
      for (; p[0].Symbol != sym; p++)
        ;
      cf                 = p[0].Freq - 1;
      s0                 = pc[0].SummFreq - pc[0].NumStats - cf;
      cf                 = 1 + ((2 * cf < s0) ? (12 * cf > s0) : 2 + cf / s0);
      ct.oneState().Freq = Min<uint>(7, cf);
    } else {
      ct.oneState().Freq = pc[0].oneState().Freq;
    }

    do {
      PPM_CONTEXT *pc1 = (PPM_CONTEXT *)AllocContext();
      if (!pc1) {
        // Signal caller to trigger pruning/recovery path
        saved_pc = pc;
        return 0;
      }
      ((uint *)pc1)[0] = ((uint *)&ct)[0];
      ((uint *)pc1)[1] = ((uint *)&ct)[1];
      pc1->iSuffix     = Ptr2Indx(pc);
      pc               = pc1;
      pps--;
      pps[0][0].iSuccessor = Ptr2Indx(pc);
    } while (pps != ps);

    return Ptr2Indx(pc);
  }

  uint ReduceOrder(STATE *p, PPM_CONTEXT *pc) {
    byte         tmp;
    STATE       *p1;
    PPM_CONTEXT *pc1       = pc;
    FoundState->iSuccessor = Ptr2Indx(pText);
    byte sym               = FoundState->Symbol;
    uint iUpBranch         = FoundState->iSuccessor;
    OrderFall++;

    if (p) {
      pc = suff(pc);
      goto LOOP_ENTRY;
    }

    while (1) {
      if (!pc->iSuffix)
        return Ptr2Indx(pc);
      pc = suff(pc);
      if (!pc) {
        printf("\n*** ReduceOrder: suff(pc) returned nullptr ***\n");

        assert(false && "ReduceOrder: null suffix context");
        return 0;
      }

      if (pc->NumStats) {
        p = getStats(pc);
        if (!p) {
          printf("\n*** ReduceOrder: getStats(pc) returned nullptr ***\n");

          assert(false && "ReduceOrder: null stats pointer");
          return 0;
        }
        for (; p[0].Symbol != sym; p++)
          ;
        tmp = 2 * (p->Freq < MAX_FREQ - 3);
        p->Freq += tmp;
        pc->SummFreq += tmp;
      } else {
        p = &(pc->oneState());
        p->Freq += (p->Freq < 11);
      }

    LOOP_ENTRY:
      if (p->iSuccessor)
        break;
      p->iSuccessor = iUpBranch;
      OrderFall++;
    }

    if (p->iSuccessor <= iUpBranch) {
      p1            = FoundState;
      FoundState    = p;
      p->iSuccessor = CreateSuccessors(0, 0, pc);
      FoundState    = p1;
    }

    if (OrderFall == 1 && pc1 == MaxContext) {
      FoundState->iSuccessor = p->iSuccessor;
      pText--;
    }

    return p->iSuccessor;
  }

  int                          PrevSuccess;
  word                         BinSumm[25][64];

  template <int ProcMode> void processBinSymbol(PPM_CONTEXT &q, int symbol) {
    STATE       &rs     = q.oneState();
    PPM_CONTEXT *q_suff = suff(&q);
    if (!q_suff)
      return;
    int i = NS2BSIndx[q_suff->NumStats] + PrevSuccess + q.Flags +
            ((RunLength >> 26) & 0x20);
    word &bs = BinSumm[QTable[rs.Freq - 1]][i];
    BSumm    = bs;
    bs -= (BSumm + 64) >> PERIOD_BITS;

    int flag = ProcMode ? 0 : rs.Symbol != symbol;

    if (flag) {
      CharMask[rs.Symbol] = EscCount;
      NumMasked           = 0;
      PrevSuccess         = 0;
      FoundState          = 0;
    } else {
      bs += INTERVAL;
      rs.Freq += (rs.Freq < 196);
      RunLength++;
      PrevSuccess = 1;
      FoundState  = &rs;
    }
  }

  template <int ProcMode> void processSymbol1(PPM_CONTEXT &q, int symbol) {
    STATE *p     = getStats(&q);

    int    cnum  = q.NumStats;
    int    i     = p[0].Symbol;
    int    low   = 0;
    int    freq  = p[0].Freq;
    int    total = q.SummFreq;
    int    flag;

    int    count = 0;

    if (ProcMode) {

      flag = count < freq;
    } else {
      flag = i == symbol;
    }

    if (flag) {

      PrevSuccess = 0;
      p[0].Freq += 4;
      q.SummFreq += 4;

    } else {

      PrevSuccess = 0;

      for (low = freq, i = 1; i <= cnum; i++) {
        freq = p[i].Freq;
        flag = ProcMode ? low + freq > count : p[i].Symbol == symbol;
        if (flag)
          break;
        low += freq;
      }

      if (flag) {
        p[i].Freq += 4;
        q.SummFreq += 4;
        if (p[i].Freq > p[i - 1].Freq)
          SWAP(p[i], p[i - 1]), i--;
        p = &p[i];
      } else {
        if (q.iSuffix)
          PrefetchData(suff(&q));
        freq      = total - low;
        NumMasked = cnum;
        for (i = 0; i <= cnum; i++)
          CharMask[p[i].Symbol] = EscCount;
        p = NULL;
      }
    }

    FoundState = p;
    if (p && (p[0].Freq > MAX_FREQ))
      FoundState = rescale(q, OrderFall, FoundState);
  }

  SEE2_CONTEXT                 SEE2Cont[23][32];
  SEE2_CONTEXT                 DummySEE2Cont;

  template <int ProcMode> void processSymbol2(PPM_CONTEXT &q, int symbol) {
    byte          px[256];
    STATE        *p = getStats(&q);

    int           c;
    int           count = 0;
    int           low;
    int           see_freq;
    int           freq;
    int           cnum = q.NumStats;

    SEE2_CONTEXT *psee2c;
    if (cnum != 0xFF) {
      PPM_CONTEXT *q_suff = suff(&q);
      if (!q_suff)
        return;
      psee2c = SEE2Cont[QTable[cnum + 3] - 4];
      psee2c += (q.SummFreq > 10 * (cnum + 1));
      psee2c += 2 * (2 * cnum < q_suff->NumStats + NumMasked) + q.Flags;
      see_freq = psee2c->getMean() + 1;

    } else {
      psee2c   = &DummySEE2Cont;
      see_freq = 1;
    }

    int flag = 0, pl;

    int i, j;
    for (i = 0, j = 0, low = 0; i <= cnum; i++) {
      c = p[i].Symbol;
      if (CharMask[c] != EscCount) {
        CharMask[c] = EscCount;
        low += p[i].Freq;
        if (ProcMode)
          px[j++] = i;
        else if (c == symbol)
          flag = 1, j = i, pl = low;
      }
    }

    int Total = see_freq + low;

    if (ProcMode) {

      flag = count < low;
    }

    if (flag) {
      if (ProcMode) {
        for (low = 0, i = 0; (low += p[j = px[i]].Freq) <= count; i++)
          ;
      } else {
        low = pl;
      }
      p += j;

      freq = p[0].Freq;

      if (see_freq > 2)
        psee2c->Summ -= see_freq;
      psee2c->update();

      FoundState = p;
      p[0].Freq += 4;
      q.SummFreq += 4;
      if (p[0].Freq > MAX_FREQ)
        FoundState = rescale(q, OrderFall, FoundState);
      RunLength = InitRL;
      EscCount++;
    } else {
      low       = Total;
      freq      = see_freq;
      NumMasked = cnum;
      psee2c->Summ += Total - see_freq;
    }
  }

  struct qsym {
    word sym;
    word freq;
    word total;

    void store(uint _sym, uint _freq, uint _total) {
      sym   = _sym;
      freq  = _freq;
      total = _total;
    }
  };

  qsym SQ[1024];
  uint SQ_ptr;

  uint sqp[256]; // Probability array for 256 possible bytes

  // Convert SQ[] frequency array to cumulative probability distribution in
  // sqp[]
  void ConvertSQ(void) {
    memset(sqp, 0, sizeof(sqp));

    uint cum = 0xFFFFFF00;
    for (uint i = 0; i < SQ_ptr; i++) {
      const qsym &sq   = SQ[i];
      const uint  c    = sq.sym;
      const uint  prob = (static_cast<qword>(cum) * sq.freq) / sq.total;

      if (c < 256) {
        sqp[c] = prob + 1;
      } else {
        cum = prob;
      }
    }
  }

  void processBinSymbol_T(PPM_CONTEXT &q) {
    STATE       &rs     = q.oneState();
    PPM_CONTEXT *q_suff = suff(&q);
    if (!q_suff)
      return;
    int i = NS2BSIndx[q_suff->NumStats] + PrevSuccess + q.Flags +
            ((RunLength >> 26) & 0x20);
    word &bs = BinSumm[QTable[rs.Freq - 1]][i];
    BSumm    = bs;

    SQ[SQ_ptr++].store(rs.Symbol, BSumm + BSumm, SCALE);
    SQ[SQ_ptr++].store(256, SCALE - BSumm - BSumm, SCALE);

    CharMask[rs.Symbol] = EscCount;
    NumMasked           = 0;
  }

  void processSymbol1_T(PPM_CONTEXT &q) {
    STATE    *p     = getStats(&q);
    const int cnum  = q.NumStats;
    const int total = q.SummFreq;

    // Store all symbols and accumulate frequencies
    int low = 0;
    for (int i = 0; i <= cnum; i++) {
      const int freq = p[i].Freq;
      SQ[SQ_ptr++].store(p[i].Symbol, freq, total);
      low += freq;
      CharMask[p[i].Symbol] = EscCount; // Mark as seen
    }

    // Store escape symbol with remaining probability
    SQ[SQ_ptr++].store(256, total - low, total);
    NumMasked = cnum;

    if (q.iSuffix)
      PrefetchData(suff(&q));
  }

  void processSymbol2_T(PPM_CONTEXT &q) {
    STATE    *p    = getStats(&q);
    const int cnum = q.NumStats;

    // Calculate SEE2 context and escape frequency
    int see_freq;
    if (cnum != 0xFF) {
      PPM_CONTEXT *q_suff = suff(&q);
      if (!q_suff)
        return;
      const int     base_idx = QTable[cnum + 3] - 4;
      SEE2_CONTEXT *psee2c   = SEE2Cont[base_idx];
      const int     adj1     = (q.SummFreq > 10 * (cnum + 1));
      const int adj2 = 2 * (2 * cnum < q_suff->NumStats + NumMasked) + q.Flags;
      see_freq       = psee2c[adj1 + adj2].getMean() + 1;
    } else {
      see_freq = 1;
    }

    // Process unmasked symbols (not seen in longer contexts)
    int        low      = 0;
    const uint curr_esc = EscCount;
    qsym      *sq_write = &SQ[SQ_ptr];

    for (int i = 0; i <= cnum; i++) {
      const int c = p[i].Symbol;
      if (CharMask[c] != curr_esc) {
        const int freq = p[i].Freq;
        low += freq;
        sq_write->store(c, freq, 0); // Total filled in fixup pass
        sq_write++;
        CharMask[c] = curr_esc;
      }
    }

    // Fixup pass: set Total for all stored symbols
    const int Total = see_freq + low;
    for (qsym *sq = &SQ[SQ_ptr]; sq < sq_write; sq++) {
      sq->total = Total;
    }

    SQ_ptr = sq_write - SQ;
    SQ[SQ_ptr++].store(256, see_freq, Total);
    NumMasked = cnum;
  }

  uint cxt;
  uint y;
  unsigned long long counter_; // Per-instance byte counter for statistics

  // Initialize PPMD model with specified parameters
  // MaxOrder: Maximum context length (typically 25)
  // MMAX: Memory in MB (will be shifted: MMAX << 20 bytes)
  // CutOff: Whether to use memory cutoff (1) or full reset (0)
  // filesize: Expected file size (unused in this version)
  uint Init(uint MaxOrder, uint MMAX, uint CutOff, uint filesize) {
    _MaxOrder = MaxOrder;
    _CutOff   = CutOff;
    _MMAX     = MMAX;
    _filesize = filesize;
    counter_  = 0; // Initialize counter for this instance

    PPMD_STARTUP(); // Initialize lookup tables

    if (!StartSubAllocator(_MMAX)) // Allocate heap (MMAX << 20 bytes!)
      return 1;

    StartModelRare(); // Create initial context tree

    cxt = 0;
    y   = 1;

    return 0;
  }

  ~ppmd_Model() { StopSubAllocator(); }

  // Generate probability distribution for next byte
  // Uses escape mechanism: tries longest context first, falls back to shorter
  // ones Output: sqp[] array with probabilities for all 256 possible bytes
  void ppmd_PrepareByte(void) {
    SQ_ptr                  = 0;
    NumMasked               = 0;
    int          _OrderFall = OrderFall;

    PPM_CONTEXT *MinContext = MaxContext; // Start with longest context
    if (MinContext->NumStats) {
      processSymbol1_T(MinContext[0]); // Multi-symbol context
    } else {
      processBinSymbol_T(MinContext[0]); // Binary context optimization
    }

    // Escape mechanism: try shorter and shorter contexts
    while (1) {
      do {
        if (!MinContext->iSuffix)
          goto Break; // Reached root context
        OrderFall++;
        MinContext = suff(MinContext); // Move to parent (shorter) context
      } while (MinContext->NumStats == NumMasked);
      processSymbol2_T(MinContext[0]); // Add escape predictions
    }

  Break:
    EscCount++;
    NumMasked = 0;
    OrderFall = _OrderFall;

    ConvertSQ(); // Convert internal format to probability array
  }

  // Update model after encoding/decoding a byte
  // Increments frequency, creates new contexts if needed, handles memory
  // exhaustion
  void ppmd_UpdateByte(uint c) {
    PPM_CONTEXT *MinContext = MaxContext;
    // Find symbol in current context
    if (MinContext->NumStats) {
      processSymbol1<0>(MinContext[0], c);
    } else {
      processBinSymbol<0>(MinContext[0], c);
    }

    // Search parent contexts if not found (escape)
    while (!FoundState) {
      do {

        OrderFall++;
        MinContext = suff(MinContext); // Move to shorter context
      } while (MinContext->NumStats == NumMasked);
      processSymbol2<0>(MinContext[0], c);
    }

    // Update frequency and create new contexts
    PPM_CONTEXT *p;
    PPM_CONTEXT *foundSucc = getSucc(FoundState);
    if ((OrderFall != 0) || (foundSucc == nullptr) ||
        ((byte *)foundSucc < UnitsStart)) {
      p = UpdateModel(MinContext); // Increment freq, maybe create new context
      if (p)
        MaxContext = p;
    } else {
      p = MaxContext = getSucc(FoundState); // Move to existing next context
      if (p == nullptr) {
        printf("encodeSymbol2: Invalid MaxContext (nullptr) from getSucc\n");

        return;
      }
    }

    // Handle memory exhaustion
    if (p == 0) {
      if (_CutOff) {
        printf("reset\n");
        RestoreModelRare(); // Try to reclaim memory
      } else {
        StartModelRare(); // Full model reset
      }
    }
  }
};

#pragma pack()

// PPMD public interface - wraps internal ppmd_Model
// Constructor: Creates model with specified order and memory
// Parameters:
//   order: Maximum context length (typically 25)
//   memory: Memory in MB (⚠️ will be left-shifted: memory << 20 bytes)
//   bit_context: Current byte being processed
//   vocab: Valid byte vocabulary
PPMD::PPMD(int order, int memory, const unsigned int &bit_context,
           const std::vector<bool> &vocab)
    : ByteModel(vocab), byte_(bit_context),
      byte_map_(Eigen::VectorXi::Zero(256)) {
  ppmd_model_.reset(new ppmd_Model());
  ppmd_model_->Init(order, memory, 1, 0); // memory << 20 bytes allocated!
}

PPMD::~PPMD() {}

// Called after each byte to update model and generate new predictions
// Output: probs_ contains probability distribution for next 256 possible bytes
void PPMD::ByteUpdate() {
  ++ppmd_model_->counter_;
  ppmd_model_->ppmd_UpdateByte(byte_); // Update frequencies with actual byte
  ppmd_model_->ppmd_PrepareByte();     // Generate predictions for next byte

  // Extract probabilities from internal format
  for (int i = 0; i < 256; ++i) {
    probs_[i] = ppmd_model_->sqp[i]; // Get raw frequency
    if (probs_[i] < 1)
      probs_[i] = 1; // Minimum probability (avoid zero)
  }
  ByteModel::ByteUpdate();
  probs_ /= probs_.sum(); // Normalize to sum to 1.0
}

} // namespace PPMD
