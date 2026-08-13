/* Exact category-only counter for all unordered seven-card subsets of a 52-card deck.
 * Card id mapping deliberately matches reference/cards.py: id % 13 is rank 2..A,
 * id / 13 is suit.  This is a release-gate acceleration, not semantic authority.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int straight_high(uint16_t mask) {
    for (int high = 14; high >= 6; --high) {
        uint16_t needed = (uint16_t)(0x1fU << (high - 6)); /* rank 2 is bit zero */
        if ((mask & needed) == needed) return high;
    }
    return ((mask & ((1U << 12) | 0x0fU)) == ((1U << 12) | 0x0fU)) ? 5 : 0;
}

static int category(int a, int b, int c, int d, int e, int f, int g) {
    int ids[7] = {a,b,c,d,e,f,g};
    unsigned rank_count[13] = {0}, suit_count[4] = {0};
    uint16_t rank_mask = 0, suit_mask[4] = {0};
    for (int i = 0; i < 7; ++i) {
        unsigned rank = (unsigned)(ids[i] % 13), suit = (unsigned)(ids[i] / 13);
        ++rank_count[rank]; ++suit_count[suit];
        rank_mask |= (uint16_t)(1U << rank); suit_mask[suit] |= (uint16_t)(1U << rank);
    }
    for (int suit = 0; suit < 4; ++suit)
        if (suit_count[suit] >= 5 && straight_high(suit_mask[suit])) return 8;
    int trips = 0, pairs = 0;
    for (int rank = 0; rank < 13; ++rank) {
        if (rank_count[rank] == 4) return 7;
        if (rank_count[rank] >= 3) ++trips;
        if (rank_count[rank] >= 2) ++pairs;
    }
    if (trips && (pairs >= 2 || trips >= 2)) return 6;
    for (int suit = 0; suit < 4; ++suit) if (suit_count[suit] >= 5) return 5;
    if (straight_high(rank_mask)) return 4;
    if (trips) return 3;
    if (pairs >= 2) return 2;
    if (pairs) return 1;
    return 0;
}

static int sample(void) {
    char line[128];
    while (fgets(line, sizeof line, stdin) != NULL) {
        int ids[7];
        char *cursor = line, *end;
        if (strchr(line, '\n') == NULL && !feof(stdin)) return EXIT_FAILURE;
        for (int i = 0; i < 7; ++i) {
            long value;
            while (*cursor == ' ' || *cursor == '\t') ++cursor;
            if (*cursor == '\0' || *cursor == '\n') return EXIT_FAILURE;
            errno = 0;
            value = strtol(cursor, &end, 10);
            if (errno == ERANGE || end == cursor || value < 0 || value > 51) return EXIT_FAILURE;
            ids[i] = (int)value;
            cursor = end;
        }
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        if (*cursor != '\n' && *cursor != '\0') return EXIT_FAILURE;
        for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
            if (ids[i] == ids[j]) return EXIT_FAILURE;
        printf("%d\n", category(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6]));
    }
    return ferror(stdin) ? EXIT_FAILURE : EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "--sample") == 0) return sample();
    if (argc != 1) return EXIT_FAILURE;
    uint64_t counts[9] = {0}, total = 0;
    for (int a=0;a<46;++a) for (int b=a+1;b<47;++b) for (int c=b+1;c<48;++c)
    for (int d=c+1;d<49;++d) for (int e=d+1;e<50;++e) for (int f=e+1;f<51;++f)
    for (int g=f+1;g<52;++g) { ++counts[category(a,b,c,d,e,f,g)]; ++total; }
    printf("%llu", (unsigned long long)total);
    for (int i=0;i<9;++i) printf(" %llu", (unsigned long long)counts[i]);
    putchar('\n');
    return total == UINT64_C(133784560) ? EXIT_SUCCESS : EXIT_FAILURE;
}
